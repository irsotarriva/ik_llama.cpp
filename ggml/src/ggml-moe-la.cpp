#include "ggml-moe-la.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

constexpr size_t LA_PAGE = 4096;   // also the read alignment; file sectors are 512 or 4096

inline uintptr_t page_floor(uintptr_t x) { return x & ~(uintptr_t)(LA_PAGE - 1); }
inline uintptr_t page_ceil (uintptr_t x) { return (x + LA_PAGE - 1) & ~(uintptr_t)(LA_PAGE - 1); }

// ---- platform: reserved ranges and unbuffered positional reads ----------

uint8_t * os_reserve(size_t size) {
#if defined(_WIN32)
    return (uint8_t *) VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_NOACCESS);
#else
    void * p = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : (uint8_t *) p;
#endif
}

bool os_commit(uint8_t * p, size_t size) {
#if defined(_WIN32)
    return VirtualAlloc(p, size, MEM_COMMIT, PAGE_READWRITE) != nullptr;
#else
    return mprotect(p, size, PROT_READ | PROT_WRITE) == 0;
#endif
}

void os_release(uint8_t * p, size_t size) {
#if defined(_WIN32)
    VirtualFree(p, size, MEM_DECOMMIT);
#else
    madvise(p, size, MADV_DONTNEED);
    mprotect(p, size, PROT_NONE);
#endif
}

#if defined(_WIN32)
typedef HANDLE la_fd;
const la_fd LA_BAD_FD = INVALID_HANDLE_VALUE;
la_fd os_open(const char * path) {
    return CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                       FILE_FLAG_NO_BUFFERING, nullptr);
}
bool os_pread(la_fd fd, uint8_t * dst, size_t size, uint64_t offset) {
    while (size > 0) {
        OVERLAPPED ov = {};
        ov.Offset     = (DWORD) (offset & 0xffffffffu);
        ov.OffsetHigh = (DWORD) (offset >> 32);
        DWORD want = (DWORD) std::min<size_t>(size, (size_t) 64 << 20), got = 0;
        if (!ReadFile(fd, dst, want, &got, &ov)) return GetLastError() == ERROR_HANDLE_EOF;
        if (got == 0) return true;   // end of file inside the last, padded page
        dst += got; size -= got; offset += got;
    }
    return true;
}
#else
typedef int la_fd;
const la_fd LA_BAD_FD = -1;
la_fd os_open(const char * path) {
#if defined(__linux__)
    return open(path, O_RDONLY | O_DIRECT);
#else
    int fd = open(path, O_RDONLY);
#if defined(F_NOCACHE)
    if (fd >= 0) fcntl(fd, F_NOCACHE, 1);
#endif
    return fd;
#endif
}
bool os_pread(la_fd fd, uint8_t * dst, size_t size, uint64_t offset) {
    while (size > 0) {
        ssize_t got = pread(fd, dst, size, (off_t) offset);
        if (got < 0) return false;
        if (got == 0) return true;
        dst += got; size -= (size_t) got; offset += (uint64_t) got;
    }
    return true;
}
#endif

// ---- state ----------------------------------------------------------------

struct la_tensor {
    uint8_t * data = nullptr;     // what tensor->data points to
    uint8_t * region_end = nullptr;
    size_t    stride = 0;         // bytes per expert
    uint64_t  file_off = 0;       // of data[0]
    int       file_id = 0;
};

struct la_layer {
    std::vector<uint8_t>  resident;   // per expert
    std::vector<uint64_t> last;       // clock of the last time an expert was served
    std::vector<float>    usage;      // decayed count of how often the router asked for an expert
    std::vector<la_tensor> tensors;   // real mode
    std::vector<int> pending;         // experts to read before this run of tokens ends
    std::vector<int16_t> gpu_slot;    // per expert: its VRAM slot, or -1
    int      n_gpu      = 0;
    int      n_resident = 0;
    int      cap        = 0;
    uint64_t clock      = 0;
    uint64_t run_start  = 0;
};

struct la_job { la_tensor * t; int expert; };

struct la_state {
    bool   enabled   = false;
    bool   real      = false;
    float  frac      = 1.0f;
    float  beta      = 0.0f;
    float  theta     = 0.0f;
    float  price_res = 0.1f;
    float  gpu_frac  = 0.0f;
    bool   evict_freq = true;   // evict the least requested expert, not the least recently served
    float  decay     = 0.98f;   // applied to the request counts every 16 tokens
    float  excess_ram = 0.0f;
    int    io_threads = 8;
    std::string profile_path;
    std::string stats_path;

    std::vector<float> profile;       // [n_layers * n_expert], empty when not given
    std::vector<std::unique_ptr<la_layer>> layers;
    std::vector<std::string> files;
    std::mutex mtx;

    // reader pool
    std::mutex              io_mtx;
    std::condition_variable io_work, io_done;
    std::deque<la_job>      io_queue;
    std::vector<std::thread> io_workers;
    int  io_pending = 0;
    bool io_stop = false;
    std::atomic<bool> io_failed{false};

    // counters, in expert-runs unless noted
    uint64_t gpu_served = 0;
    uint64_t events = 0, served = 0, reads = 0, exact_reads = 0, substituted = 0, dropped = 0, evicted = 0;
    std::atomic<uint64_t> read_bytes{0};
    double read_wait_s = 0, prefill_s = 0;
    int    peak_over = 0;      // most experts held beyond the budget at one time
    uint64_t prefill_bytes = 0;

    la_state() {
        const char * e = getenv("LA_ENABLE");
        enabled = e && atoi(e) != 0;
        if (!enabled) return;
        if (const char * v = getenv("LA_REAL"))           real      = atoi(v) != 0;
        if (const char * v = getenv("LA_RESIDENT_FRAC"))  frac      = (float) atof(v);
        if (const char * v = getenv("LA_BETA"))           beta      = (float) atof(v);
        if (const char * v = getenv("LA_THETA"))          theta     = (float) atof(v);
        if (const char * v = getenv("LA_PRICE_RESIDENT")) price_res = (float) atof(v);
        if (const char * v = getenv("LA_GPU_FRAC"))       gpu_frac  = (float) atof(v);
        if (const char * v = getenv("LA_EXCESS_RAM"))     excess_ram = (float) atof(v);
        if (const char * v = getenv("LA_EVICT"))          evict_freq = strcmp(v, "lru") != 0;
        if (const char * v = getenv("LA_DECAY"))          decay     = (float) atof(v);
        if (const char * v = getenv("LA_IO_THREADS"))     io_threads = std::max(1, atoi(v));
        if (const char * v = getenv("LA_PROFILE"))        profile_path = v;
        if (const char * v = getenv("LA_STATS"))          stats_path   = v;
        if (!profile_path.empty()) {
            if (FILE * f = fopen(profile_path.c_str(), "rb")) {
                fseek(f, 0, SEEK_END);
                long n = ftell(f);
                fseek(f, 0, SEEK_SET);
                profile.resize((size_t) n / sizeof(float));
                if (fread(profile.data(), sizeof(float), profile.size(), f) != profile.size()) profile.clear();
                fclose(f);
            }
            if (profile.empty()) fprintf(stderr, "moe-la: cannot read profile %s, starting cold\n", profile_path.c_str());
        }
        if (gpu_frac > 0 && profile.empty()) {
            fprintf(stderr, "moe-la: LA_GPU_FRAC needs LA_PROFILE, VRAM tier off\n");
            gpu_frac = 0;
        }
        if (gpu_frac > 0) fprintf(stderr, "moe-la: VRAM tier, gpu_frac=%g excess_ram=%g\n", (double) gpu_frac, (double) excess_ram);
        fprintf(stderr, "moe-la: enabled (%s), resident_frac=%g beta=%g theta=%g price_resident=%g profile=%s\n",
                real ? "real" : "modelled", (double) frac, (double) beta, (double) theta, (double) price_res,
                profile.empty() ? "(none)" : profile_path.c_str());
    }

    ~la_state() {
        report();
        {
            std::lock_guard<std::mutex> lock(io_mtx);
            io_stop = true;
            io_work.notify_all();
        }
        for (auto & w : io_workers) if (w.joinable()) w.join();
    }

    void report() {
        if (!enabled || events == 0) return;
        const double ev = (double) events;
        const double gib = 1024.0*1024.0*1024.0;
        fprintf(stderr, "moe-la: events=%llu served/event=%.3f reads/event=%.3f exact_reads/event=%.3f "
                        "substituted/event=%.3f dropped/event=%.3f\n",
                (unsigned long long) events, served/ev, reads/ev, exact_reads/ev, substituted/ev, dropped/ev);
        if (gpu_frac > 0) fprintf(stderr, "moe-la: gpu_served/event=%.3f\n", gpu_served/ev);
        if (real) {
            fprintf(stderr, "moe-la: read %.2f GiB in %.2f s of waiting (%.2f GiB/s), %llu evictions; "
                            "prefill %.2f GiB in %.2f s\n",
                    read_bytes.load()/gib, read_wait_s, read_wait_s > 0 ? read_bytes.load()/gib/read_wait_s : 0.0,
                    (unsigned long long) evicted, prefill_bytes/gib, prefill_s);
            fprintf(stderr, "moe-la: at most %d experts over the budget at one time\n", peak_over);
        }
        if (!stats_path.empty()) {
            if (FILE * f = fopen(stats_path.c_str(), "a")) {
                fprintf(f, "{\"real\":%d,\"resident_frac\":%g,\"beta\":%g,\"theta\":%g,\"price_resident\":%g,"
                           "\"events\":%llu,\"served\":%llu,\"reads\":%llu,\"exact_reads\":%llu,\"substituted\":%llu,"
                           "\"dropped\":%llu,\"evicted\":%llu,\"read_bytes\":%llu,\"read_wait_s\":%.4f,"
                           "\"prefill_bytes\":%llu,\"prefill_s\":%.4f}\n",
                        real ? 1 : 0, (double) frac, (double) beta, (double) theta, (double) price_res,
                        (unsigned long long) events, (unsigned long long) served, (unsigned long long) reads,
                        (unsigned long long) exact_reads, (unsigned long long) substituted,
                        (unsigned long long) dropped, (unsigned long long) evicted,
                        (unsigned long long) read_bytes.load(), read_wait_s,
                        (unsigned long long) prefill_bytes, prefill_s);
                fclose(f);
            }
        }
    }

    la_layer * fresh(int n_expert) {
        la_layer * L = new la_layer;
        L->resident.assign(n_expert, 0);
        L->last.assign(n_expert, 0);
        L->usage.assign(n_expert, 0.0f);
        L->gpu_slot.assign(n_expert, -1);
        return L;
    }

    // Residency plan from the profile: rank every (layer, expert) by frequency,
    // the top frac share starts resident and fixes the capacity of each layer.
    void plan(int n_expert) {
        const int n_layers = (int) (profile.size() / (size_t) n_expert);
        std::vector<int> order(profile.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return profile[a] > profile[b]; });
        const size_t n_gpu = (size_t) std::min<double>((double) order.size(), (double) gpu_frac * (double) order.size());
        const size_t keep = (size_t) std::min<double>((double) order.size(), (double) n_gpu + (double) frac * (double) order.size());
        layers.resize(n_layers);
        for (int l = 0; l < n_layers; ++l) layers[l].reset(fresh(n_expert));
        for (size_t i = 0; i < n_gpu; ++i) {
            la_layer & L = *layers[order[i] / n_expert];
            L.gpu_slot[order[i] % n_expert] = (int16_t) L.n_gpu++;
        }
        for (size_t i = n_gpu; i < keep; ++i) {
            la_layer & L = *layers[order[i] / n_expert];
            L.resident[order[i] % n_expert] = 1;
            L.n_resident++;
            L.cap++;
        }
    }

    la_layer & get(int layer, int n_expert) {
        if (layers.empty() && !profile.empty() && profile.size() % (size_t) n_expert == 0) plan(n_expert);
        if ((int) layers.size() <= layer) layers.resize(layer + 1);
        if (!layers[layer]) {
            layers[layer].reset(fresh(n_expert));
            layers[layer]->cap = (int) std::min<double>(n_expert, (double) frac * n_expert + 0.5);
        }
        return *layers[layer];
    }

    // ---- real mode I/O ----

    static void expert_span(const la_tensor & t, int e, uint8_t *& lo, uint8_t *& hi) {
        const uintptr_t s = (uintptr_t) t.data + (uintptr_t) e * t.stride;
        lo = (uint8_t *) page_floor(s);
        hi = (uint8_t *) std::min<uintptr_t>(page_ceil(s + t.stride), (uintptr_t) t.region_end);
    }

    void worker() {
        std::vector<la_fd> fds;
        for (;;) {
            la_job job;
            {
                std::unique_lock<std::mutex> lock(io_mtx);
                io_work.wait(lock, [&] { return io_stop || !io_queue.empty(); });
                if (io_stop) break;
                job = io_queue.front();
                io_queue.pop_front();
            }
            const la_tensor & t = *job.t;
            if ((int) fds.size() <= t.file_id) fds.resize(t.file_id + 1, LA_BAD_FD);
            if (fds[t.file_id] == LA_BAD_FD) fds[t.file_id] = os_open(files[t.file_id].c_str());
            uint8_t * lo, * hi;
            expert_span(t, job.expert, lo, hi);
            const size_t n = (size_t) (hi - lo);
            // lo may sit a little before data (then it holds the tail of the tensor stored before this one)
            const uint64_t off = (uint64_t) ((int64_t) t.file_off + (int64_t) (lo - t.data));
            bool ok = fds[t.file_id] != LA_BAD_FD && os_commit(lo, n) && os_pread(fds[t.file_id], lo, n, off);
            if (!ok) io_failed.store(true);
            read_bytes.fetch_add(n);
            {
                std::lock_guard<std::mutex> lock(io_mtx);
                if (--io_pending == 0) io_done.notify_all();
            }
        }
    }

    void read_experts(la_layer & L, const std::vector<int> & experts) {
        if (L.tensors.empty() || experts.empty()) return;
        std::unique_lock<std::mutex> lock(io_mtx);
        if (io_workers.empty()) {
            for (int i = 0; i < io_threads; ++i) io_workers.emplace_back([this] { worker(); });
        }
        for (int e : experts) {
            for (auto & t : L.tensors) {
                io_queue.push_back({ &t, e });
                io_pending++;
            }
        }
        io_work.notify_all();
        io_done.wait(lock, [&] { return io_pending == 0; });
        if (io_failed.load()) {
            fprintf(stderr, "moe-la: reading an expert failed\n");
            abort();
        }
    }

    // true if a should be evicted before b
    bool colder(const la_layer & L, int a, int b) const {
        if (evict_freq && L.usage[a] != L.usage[b]) return L.usage[a] < L.usage[b];
        return L.last[a] < L.last[b];
    }

    // evict the coldest experts until the layer is within its budget
    void trim(la_layer & L) {
        const int n_expert = (int) L.resident.size();
        while (L.n_resident > L.cap) {
            int victim = -1;
            for (int x = 0; x < n_expert; ++x) {
                if (L.resident[x] && (victim < 0 || colder(L, x, victim))) victim = x;
            }
            if (victim < 0) break;
            L.resident[victim] = 0;
            L.n_resident--;
            if (real) release_expert(L, victim);
        }
    }

    void release_expert(la_layer & L, int e) {
        for (auto & t : L.tensors) {
            // keep the pages shared with the neighbouring experts
            const uintptr_t s  = (uintptr_t) t.data + (uintptr_t) e * t.stride;
            const uintptr_t lo = page_ceil(s), hi = page_floor(s + t.stride);
            if (hi > lo) os_release((uint8_t *) lo, (size_t) (hi - lo));
        }
        evicted++;
    }
};

la_state & state() {
    static la_state s;
    return s;
}

inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace

bool ggml_moe_la_enabled(void) { return state().enabled; }
bool ggml_moe_la_real(void)    { return state().enabled && state().real; }

bool ggml_moe_la_register(int layer, struct ggml_tensor * tensor, const char * path, uint64_t file_offset) {
    la_state & s = state();
    if (!s.enabled || !s.real || !tensor || !tensor->data) return false;
    const int n_expert = (int) tensor->ne[2];
    const size_t nbytes = ggml_nbytes(tensor);
    if (n_expert <= 1 || nbytes % (size_t) n_expert != 0) return false;

    std::lock_guard<std::mutex> lock(s.mtx);
    la_layer & L = s.get(layer, n_expert);
    const size_t lead = (size_t) (file_offset % LA_PAGE);
    const size_t region_size = page_ceil(lead + nbytes);
    uint8_t * region = os_reserve(region_size);
    if (!region) {
        fprintf(stderr, "moe-la: cannot reserve %zu bytes for %s\n", region_size, ggml_get_name(tensor));
        return false;
    }
    la_tensor t;
    t.data = region + lead;                 // same offset inside a page as in the file
    t.region_end = region + region_size;
    t.stride = nbytes / (size_t) n_expert;
    t.file_off = file_offset;
    auto it = std::find(s.files.begin(), s.files.end(), std::string(path));
    t.file_id = (int) (it - s.files.begin());
    if (it == s.files.end()) s.files.push_back(path);
    L.tensors.push_back(t);
    tensor->data = t.data;
    return true;
}

void ggml_moe_la_prefill(void) {
    la_state & s = state();
    if (!s.enabled || !s.real) return;
    std::lock_guard<std::mutex> lock(s.mtx);
    const double t0 = now_s();
    const uint64_t b0 = s.read_bytes.load();
    int n_tensors = 0, n_experts = 0;
    for (auto & Lp : s.layers) {
        if (!Lp || Lp->tensors.empty()) continue;
        n_tensors += (int) Lp->tensors.size();
        std::vector<int> want;
        for (int e = 0; e < (int) Lp->resident.size(); ++e) if (Lp->resident[e]) want.push_back(e);
        n_experts += (int) want.size();
        s.read_experts(*Lp, want);
    }
    s.prefill_s = now_s() - t0;
    s.prefill_bytes = s.read_bytes.load() - b0;
    s.read_bytes.store(b0);
    fprintf(stderr, "moe-la: took over %d expert tensors; %d experts resident at start (%.2f GiB read in %.1f s)\n",
            n_tensors, n_experts, s.prefill_bytes/(1024.0*1024.0*1024.0), s.prefill_s);
}

void ggml_moe_la_begin(int layer) {
    la_state & s = state();
    std::lock_guard<std::mutex> lock(s.mtx);
    // A run of many tokens (a prompt batch) may take a layer over its budget, because every
    // expert the run uses has to be in memory at once. The kernels of the other layers have
    // finished by now, so bring those back under budget here: the excess is then never more
    // than what one layer needs for one batch.
    int over = 0;
    for (int l = 0; l < (int) s.layers.size(); ++l) {
        if (!s.layers[l]) continue;
        over += std::max(0, s.layers[l]->n_resident - s.layers[l]->cap);
        if (l != layer) s.trim(*s.layers[l]);
    }
    s.peak_over = std::max(s.peak_over, over);
    if ((int) s.layers.size() <= layer || !s.layers[layer]) return;
    la_layer & L = *s.layers[layer];
    s.trim(L);
    L.run_start = L.clock + 1;
    L.pending.clear();
}

void ggml_moe_la_select(int layer, const float * probs, int n_expert, int k, int32_t * ids) {
    la_state & s = state();
    std::lock_guard<std::mutex> lock(s.mtx);
    la_layer & L = s.get(layer, n_expert);

    static thread_local std::vector<int>   idx;
    static thread_local std::vector<float> adj;
    idx.resize(n_expert);
    std::iota(idx.begin(), idx.end(), 0);

    // the router top-k, most probable first
    const int kk = std::min(k, 64);
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                      [&](int a, int b) { return probs[a] > probs[b] || (probs[a] == probs[b] && a < b); });
    int wanted[64];
    for (int j = 0; j < kk; ++j) wanted[j] = idx[j];

    int picked[64];
    if (s.beta > 0) {
        const float rng = std::max(probs[wanted[0]] - probs[wanted[kk - 1]], 1e-12f);
        adj.resize(n_expert);
        for (int e = 0; e < n_expert; ++e) {
            const float excess = L.gpu_slot[e] >= 0 ? 0.0f : L.resident[e] ? s.excess_ram : 1.0f;
            adj[e] = probs[e] - s.beta * rng * excess;
        }
        std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                          [&](int a, int b) { return adj[a] > adj[b] || (adj[a] == adj[b] && a < b); });
        for (int j = 0; j < kk; ++j) picked[j] = idx[j];
        std::sort(picked, picked + kk,
                  [&](int a, int b) { return probs[a] > probs[b] || (probs[a] == probs[b] && a < b); });
    } else {
        memcpy(picked, wanted, sizeof(int) * kk);
    }

    int n_served = 0;
    const float top = probs[picked[0]];
    for (int j = 0; j < kk; ++j) {
        const int e = picked[j];
        const float price = (L.resident[e] || L.gpu_slot[e] >= 0) ? s.price_res : 1.0f;
        if (j > 0 && s.theta > 0 && probs[e] < s.theta * top * price) {
            s.dropped++;
            continue;
        }
        ids[n_served++] = e;
    }
    for (int j = n_served; j < n_expert; ++j) ids[j] = -1;

    // accounting
    s.events++;
    s.served += n_served;
    for (int j = 0; j < kk; ++j) {
        if (!L.resident[wanted[j]] && L.gpu_slot[wanted[j]] < 0) s.exact_reads++;
        bool in_wanted = false;
        for (int i = 0; i < kk; ++i) in_wanted |= (wanted[i] == picked[j]);
        if (!in_wanted) s.substituted++;
    }

    // residency: a served expert that is not resident is read and kept; the least
    // recently served expert that this run of tokens does not use makes room
    if (L.run_start == 0) L.run_start = L.clock + 1;
    L.clock++;
    for (int j = 0; j < kk; ++j) L.usage[wanted[j]] += 1.0f;
    if (s.decay < 1.0f && (L.clock & 15) == 0) {
        for (float & u : L.usage) u *= s.decay;
    }
    for (int j = 0; j < n_served; ++j) L.last[ids[j]] = L.clock;
    for (int j = 0; j < n_served; ++j) {
        const int e = ids[j];
        if (L.gpu_slot[e] >= 0) { s.gpu_served++; continue; }
        if (L.resident[e]) continue;
        s.reads++;
        if (L.n_resident >= L.cap) {
            int victim = -1;
            for (int x = 0; x < n_expert; ++x) {
                if (!L.resident[x] || L.last[x] >= L.run_start) continue;
                if (victim < 0 || s.colder(L, x, victim)) victim = x;
            }
            if (victim >= 0) {
                L.resident[victim] = 0;
                L.n_resident--;
                if (s.real) s.release_expert(L, victim);
            } else if (!s.real && L.cap <= 0) {
                continue;   // modelled, no memory at all: read and forget
            }
        }
        L.resident[e] = 1;
        L.n_resident++;
        L.pending.push_back(e);
    }
}

void ggml_moe_la_end(int layer) {
    la_state & s = state();
    std::lock_guard<std::mutex> lock(s.mtx);
    if ((int) s.layers.size() <= layer || !s.layers[layer]) return;
    la_layer & L = *s.layers[layer];
    if (s.real && !L.pending.empty()) {
        const double t0 = now_s();
        s.read_experts(L, L.pending);
        s.read_wait_s += now_s() - t0;
    }
    L.pending.clear();
    L.run_start = L.clock + 1;
}
bool ggml_moe_la_gpu(void) { return state().enabled && state().gpu_frac > 0; }

int ggml_moe_la_gpu_plan(int layer, int n_expert, int32_t * experts, int max_slots) {
    la_state & s = state();
    if (!s.enabled || s.gpu_frac <= 0) return 0;
    std::lock_guard<std::mutex> lock(s.mtx);
    la_layer & L = s.get(layer, n_expert);
    int n = 0;
    for (int e = 0; e < n_expert; ++e) {
        const int slot = L.gpu_slot[e];
        if (slot < 0) continue;
        if (slot >= max_slots) { L.gpu_slot[e] = -1; continue; }
        experts[slot] = e;
        n = std::max(n, slot + 1);
    }
    return n;
}

// Called when the engine could not create the slots of a layer: its experts go back to the CPU.
void ggml_moe_la_gpu_cancel(int layer) {
    la_state & s = state();
    std::lock_guard<std::mutex> lock(s.mtx);
    if ((int) s.layers.size() <= layer || !s.layers[layer]) return;
    la_layer & L = *s.layers[layer];
    std::fill(L.gpu_slot.begin(), L.gpu_slot.end(), (int16_t) -1);
    L.n_gpu = 0;
}

void ggml_moe_la_aux(int layer, int kind, const int32_t * ids, int k, void * out) {
    la_state & s = state();
    const la_layer * L = (int) s.layers.size() > layer ? s.layers[layer].get() : nullptr;
    for (int j = 0; j < k; ++j) {
        const int e = ids[j];
        const int slot = (L && e >= 0 && e < (int) L->gpu_slot.size()) ? L->gpu_slot[e] : -1;
        switch (kind) {
            case 1: ((int32_t *) out)[j] = slot >= 0 ? slot : (L ? L->n_gpu : 0); break;   // else the all-zero slot
            case 2: ((int32_t *) out)[j] = slot >= 0 ? -1 : e;   break;
            default: ((float *) out)[j]  = slot >= 0 ? 1.0f : 0.0f; break;
        }
    }
}
