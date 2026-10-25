#include "ggml-moe-la.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
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
#if defined(__APPLE__)
#include <mach/mach.h>
#endif
#endif

namespace {

// The memory page, also the read alignment (file sectors are 512 or 4096). Not a constant:
// 16 KiB on Apple Silicon, and mprotect refuses anything that is not a whole page.
#if defined(_WIN32)
const size_t LA_PAGE = 4096;
#else
const size_t LA_PAGE = (size_t) sysconf(_SC_PAGESIZE);
#endif

inline uintptr_t page_floor(uintptr_t x) { return x & ~(uintptr_t)(LA_PAGE - 1); }
inline uintptr_t page_ceil (uintptr_t x) { return (x + LA_PAGE - 1) & ~(uintptr_t)(LA_PAGE - 1); }

// ---- platform: reserved ranges and unbuffered positional reads ----------

#if defined(__linux__)
// On Linux every change of protection inside a range splits its mapping, and a process may hold
// 65,530 mappings (vm.max_map_count; a container cannot raise it). A model with 48 layers of 512
// experts in three tensors has 73,728 expert ranges, so a half-resident cache runs out and mprotect
// fails. The ranges are therefore mapped readable once and only their pages are given back.
// LA_GUARD=1 brings the protection back: an expert that is not resident then faults when read.
const bool LA_GUARD = getenv("LA_GUARD") && atoi(getenv("LA_GUARD")) != 0;
#endif

// LA_WIRE=1 (not Windows): lock the pages of resident experts in memory. macOS compresses and swaps
// anonymous memory it is short of, and an expert it has compressed is as slow to use as one read
// from the disk; the CPU-only real cache generated at 0.04 tok/s on the 8 GB M1 against 0.31 stock,
// which nothing measured explained. MLX wires its memory for the same reason (mx.set_wired_limit).
// A lock that fails is counted, not fatal (Linux allows 8 MiB by default: ulimit -l).
const bool LA_WIRE = getenv("LA_WIRE") && atoi(getenv("LA_WIRE")) != 0;
std::atomic<uint64_t> g_wire_fail{0};
// time spent in os_commit and os_release, all threads together: each is a system call per expert
std::atomic<uint64_t> g_os_ns{0}, g_os_calls{0};

struct os_timer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    ~os_timer() {
        g_os_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        g_os_calls++;
    }
};

uint8_t * os_reserve(size_t size) {
#if defined(_WIN32)
    return (uint8_t *) VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_NOACCESS);
#elif defined(__linux__)
    void * p = LA_GUARD ? mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
                        : mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? nullptr : (uint8_t *) p;
#else
    void * p = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : (uint8_t *) p;
#endif
}

bool os_commit(uint8_t * p, size_t size) {
    os_timer timer;
#if defined(_WIN32)
    return VirtualAlloc(p, size, MEM_COMMIT, PAGE_READWRITE) != nullptr;
#else
#if defined(__linux__)
    const bool ok = !LA_GUARD || mprotect(p, size, PROT_READ | PROT_WRITE) == 0;
#else
    const bool ok = mprotect(p, size, PROT_READ | PROT_WRITE) == 0;
#endif
    if (ok && LA_WIRE && mlock(p, size) != 0) {
        g_wire_fail++;
    }
    return ok;
#endif
}

void os_release(uint8_t * p, size_t size) {
    os_timer timer;
#if !defined(_WIN32)
    if (LA_WIRE) {
        munlock(p, size);
    }
#endif
#if defined(_WIN32)
    VirtualFree(p, size, MEM_DECOMMIT);
#elif defined(__APPLE__)
    // MADV_DONTNEED keeps anonymous pages on macOS: they are compressed and swapped instead of
    // dropped. Mapping fresh pages over the range is what actually gives the memory back.
    mmap(p, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
#else
    madvise(p, size, MADV_DONTNEED);
#if defined(__linux__)
    if (!LA_GUARD) return;
#endif
    mprotect(p, size, PROT_NONE);
#endif
}

// Memory the system could give a program now, in bytes; 0 if unknown.
uint64_t os_available() {
#if defined(_WIN32)
    MEMORYSTATUSEX ms = {};
    ms.dwLength = sizeof(ms);
    return GlobalMemoryStatusEx(&ms) ? (uint64_t) ms.ullAvailPhys : 0;
#elif defined(__APPLE__)
    // free, inactive (file cache and pages not touched lately) and purgeable pages. Compressed pages
    // are not counted: they are memory already taken.
    vm_statistics64_data_t vm;
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vm, &n) != KERN_SUCCESS) return 0;
    return ((uint64_t) vm.free_count + vm.inactive_count + vm.purgeable_count) * (uint64_t) vm_page_size;
#elif defined(__linux__)
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    uint64_t kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemAvailable: %llu kB", (unsigned long long *) &kb) == 1) break;
    }
    fclose(f);
    return kb * 1024;
#else
    return 0;
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
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) fd = open(path, O_RDONLY);   // some filesystems refuse O_DIRECT: fall back to the page cache
    return fd;
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

inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

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
    std::vector<uint32_t> asked;      // plain count of the same, for LA_PROFILE_OUT
    std::vector<la_tensor> tensors;   // real mode
    std::vector<int> pending;         // experts to read before this run of tokens ends
    std::vector<int16_t> gpu_slot;    // per expert: its VRAM slot, or -1
    struct ggml_tensor * transit[3] = { nullptr, nullptr, nullptr };   // GPU scratch for prompt batches (gate, up, down)
    struct ggml_tensor * slots[3]   = { nullptr, nullptr, nullptr };   // the slot tensors, for LA_GPU_ADAPT
    const uint8_t *      host[3]    = { nullptr, nullptr, nullptr };   // the expert tensors as the loader mapped them
    size_t               host_stride[3] = { 0, 0, 0 };
    // background loading: experts the router asked for that were substituted away
    std::vector<uint8_t> bg;                        // per expert: 1 while a background read is wanted or running
    std::vector<std::pair<float, int>> bg_want;     // this run of tokens: (probability, expert)
    std::vector<int> bg_done;                       // read, waiting to be admitted (real: guarded by io_mtx)
    std::vector<int> ahead;                         // experts being read ahead for the run about to start
    int      ahead_pending = 0;                     // their read jobs still queued or running (guarded by io_mtx)
    float    bg_credit  = 0;                        // modelled mode: reads this layer may still start
    int      n_gpu      = 0;
    int      n_resident = 0;
    int      cap        = 0;
    uint64_t clock      = 0;
    uint64_t run_start  = 0;
};

// `count` consecutive experts of one tensor, read with one request. owner: the layer that is reading
// ahead, null for a read somebody is waiting for.
struct la_job { la_tensor * t; int expert; int count; la_layer * owner; };
struct la_bg_job { la_layer * L; int expert; };

struct la_state {
    bool   enabled   = false;
    bool   real      = false;
    float  frac      = 1.0f;
    float  beta      = 0.0f;
    float  gamma     = 0.0f;    // multiplicative bias: a non-resident expert competes with p * exp(-gamma)
    int    prefetch  = 0;       // experts per token read ahead for the next layer
    int    protect   = 0;       // the router's first `protect` experts are always served
    float  wcap      = 0.0f;    // > 0: substitute only while the routing weight given up stays under this share
    float  theta     = 0.0f;
    float  price_res = 0.1f;
    float  gpu_frac  = 0.0f;
    bool   async      = false;   // load substituted-away experts in the background
    float  async_rate = 1.5f;    // modelled mode: background reads per layer per token
    int    async_queue = 64;     // real mode: most experts waiting for a background read
    int    async_threads = 2;
    bool   evict_freq = true;   // evict the least requested expert, not the least recently served
    float  decay     = 0.98f;   // applied to the request counts every 16 tokens
    float  excess_ram = 0.0f;
    int    io_threads = 8;
    int    ahead_k   = -1;      // batches: experts per token guessed for the next layer and read ahead; -1 = as many as a token uses, 0 = off
    size_t read_span = (size_t) 8 << 20;   // most bytes of neighbouring experts read with one request
    int    last_k    = 0;
    int    last_layer = -1;    // the last layer whose experts the cache holds (real mode)
    // VRAM tier, batches of at least this many tokens (0 = never). A GPU build already sends the expert
    // products of a big batch to the GPU when the experts are whole in host memory, at the price of
    // copying the layer's experts to the card for every batch. Without the real cache such a batch
    // takes that route instead of the tier's split. RTX 5090, 8 CPU threads, 2,986-token prompt, 50%
    // and 80% of the experts in slots: one batch 1,885 and 1,817 tok/s against 940 and 1,436 for the
    // split; batches of 1,024 749 and 741 against 858 and 1,443. So it pays from about 1,200 tokens
    // at 50% and 2,200 at 80%, hence the default.
    // With the real cache the experts are only partly in memory and that route is closed; the scratch
    // tensors replace it, but only when LA_GPU_BATCH is given: they were 19% slower than the CPU side
    // in batches of 1,024 and 3 to 18% faster with the prompt in one batch.
    int    gpu_batch = 2048;
    bool   gpu_batch_given = false;
    int    gpu_adapt = 0;       // VRAM tier: every this many generated tokens a layer may swap one slot; 0 = the slots stay as loaded
    float  adapt_margin = 1.5f; // ... if the expert coming in is asked for this many times as often as the one going out
    uint64_t swaps = 0;
    double t_select = 0, t_aux = 0;     // seconds spent in the selection and in splitting it between the sides
    uint64_t transit_bytes = 0, transit_runs = 0;
    double   transit_s = 0;
    uint64_t pass_transit0 = 0;
    double   pass_transit_s0 = 0;
    std::string profile_path;
    std::string stats_path;
    std::string profile_out;          // LA_PROFILE_OUT: write what the router asked for, as a profile

    std::vector<float> profile;       // [n_layers * n_expert], empty when not given
    std::vector<std::unique_ptr<la_layer>> layers;
    std::vector<std::string> files;
    std::vector<const void *> owned;   // data pointers of the tensors taken over in real mode
    std::mutex mtx;

    // reader pool
    std::mutex              io_mtx;
    std::condition_variable io_work, io_done;
    std::deque<la_job>      io_queue;
    std::deque<la_job>      ahead_queue;      // served when io_queue is empty
    std::vector<std::thread> io_workers;
    int  io_pending = 0;
    bool io_stop = false;
    std::condition_variable  bg_work;
    std::deque<la_bg_job>    bg_queue;
    std::vector<std::thread> bg_workers;
    uint64_t bg_requested = 0, bg_admitted = 0, bg_skipped = 0;
    uint64_t pf_requested = 0, pf_already = 0;
    std::atomic<uint64_t> bg_bytes{0};
    std::atomic<bool> io_failed{false};

    // counters, in expert-runs unless noted
    uint64_t gpu_served = 0;
    uint64_t events = 0, served = 0, reads = 0, exact_reads = 0, substituted = 0, dropped = 0, evicted = 0;
    std::atomic<uint64_t> read_bytes{0};
    double read_wait_s = 0, prefill_s = 0;
    int    peak_over = 0;      // most experts held beyond the budget at one time
    std::atomic<uint64_t> ahead_bytes{0};
    uint64_t ahead_experts = 0, ahead_unused = 0;
    // one pass over the layers (a batch of tokens), for the log line of long batches
    double   pass_t0 = 0, pass_wait0 = 0;
    uint64_t pass_bytes0 = 0, pass_ahead0 = 0, pass_unused0 = 0, pass_reads0 = 0;
    uint64_t prefill_bytes = 0;

    // Memory by observation (real mode, LA_HEADROOM_MB > 0). The resident share is set once at start,
    // from the memory free then; other programs may want memory later. A watcher reads the memory the
    // system has available every LA_HEADROOM_MS. Below the reserve it lowers the budgets of the layers
    // and gives the coldest experts back at once; with room to spare again it raises them, never past
    // the budgets of the start, and the experts come back as they are asked for. After Magnitude
    // (design/inference/engine/memory.md), which keeps a reserve of the larger of 10% and 2 GiB.
    uint64_t headroom = 0;              // bytes kept free for the system and other programs; 0 = off
    int      headroom_ms = 100;
    float    headroom_floor = 0.05f;    // a layer keeps at least this share of its starting budget
    std::vector<int> cap0;              // the budgets at start
    int      last_end_layer = -1;       // its experts may still be in use by its kernel
    uint64_t headroom_low = UINT64_MAX; // least memory available seen
    uint64_t headroom_given = 0, headroom_taken = 0;   // expert budgets lowered and raised again, in experts
    std::thread headroom_thread;
    std::atomic<bool> headroom_stop{false};

    la_state() {
        const char * e = getenv("LA_ENABLE");
        enabled = e && atoi(e) != 0;
        if (!enabled) return;
        if (const char * v = getenv("LA_REAL"))           real      = atoi(v) != 0;
        if (const char * v = getenv("LA_RESIDENT_FRAC"))  frac      = (float) atof(v);
        if (const char * v = getenv("LA_BETA"))           beta      = (float) atof(v);
        if (const char * v = getenv("LA_GAMMA"))          gamma     = (float) atof(v);
        if (const char * v = getenv("LA_PREFETCH"))       prefetch  = atoi(v);
        if (const char * v = getenv("LA_PROTECT"))        protect   = atoi(v);
        if (const char * v = getenv("LA_WCAP"))           wcap      = std::min(1.0f, std::max(0.0f, (float) atof(v)));
        if (const char * v = getenv("LA_THETA"))          theta     = (float) atof(v);
        if (const char * v = getenv("LA_PRICE_RESIDENT")) price_res = (float) atof(v);
        if (const char * v = getenv("LA_GPU_FRAC"))       gpu_frac  = (float) atof(v);
        if (const char * v = getenv("LA_EXCESS_RAM"))     excess_ram = (float) atof(v);
        if (const char * v = getenv("LA_ASYNC"))          async     = atoi(v) != 0;
        if (const char * v = getenv("LA_ASYNC_RATE"))     async_rate = (float) atof(v);
        if (const char * v = getenv("LA_ASYNC_QUEUE"))    async_queue = std::max(1, atoi(v));
        if (const char * v = getenv("LA_ASYNC_THREADS"))  async_threads = std::max(1, atoi(v));
        if (const char * v = getenv("LA_EVICT"))          evict_freq = strcmp(v, "lru") != 0;
        if (const char * v = getenv("LA_DECAY"))          decay     = (float) atof(v);
        if (const char * v = getenv("LA_IO_THREADS"))     io_threads = std::max(1, atoi(v));
        if (const char * v = getenv("LA_AHEAD"))          ahead_k   = atoi(v);
        if (const char * v = getenv("LA_GPU_BATCH"))    { gpu_batch = atoi(v); gpu_batch_given = true; }
        if (const char * v = getenv("LA_GPU_ADAPT"))      gpu_adapt = atoi(v);
        if (const char * v = getenv("LA_GPU_ADAPT_MARGIN")) adapt_margin = (float) atof(v);
        if (const char * v = getenv("LA_READ_MB"))        read_span = (size_t) std::max(0.0, atof(v) * 1048576.0);
        if (const char * v = getenv("LA_PROFILE"))        profile_path = v;
        if (const char * v = getenv("LA_STATS"))          stats_path   = v;
        if (const char * v = getenv("LA_PROFILE_OUT"))    profile_out  = v;
        if (const char * v = getenv("LA_HEADROOM_MB"))    headroom = (uint64_t) std::max(0.0, atof(v) * 1048576.0);
        if (const char * v = getenv("LA_HEADROOM_MS"))    headroom_ms = std::max(10, atoi(v));
        if (const char * v = getenv("LA_HEADROOM_FLOOR")) headroom_floor = std::min(1.0f, std::max(0.0f, (float) atof(v)));
        if (headroom > 0 && real) {
            if (os_available() == 0) {
                fprintf(stderr, "moe-la: LA_HEADROOM_MB: this system does not say how much memory is available, off\n");
                headroom = 0;
            } else {
                fprintf(stderr, "moe-la: keeping %.0f MiB available, checked every %d ms\n", headroom / 1048576.0, headroom_ms);
                headroom_thread = std::thread([this] { watch_headroom(); });
            }
        }
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
        fprintf(stderr, "moe-la: gamma=%g protect=%d wcap=%g%s\n", (double) gamma, protect, (double) wcap,
                wcap > 0 && (beta > 0 || gamma > 0) ? " (wcap replaces beta and gamma)" : "");
        fprintf(stderr, "moe-la: enabled (%s), resident_frac=%g beta=%g theta=%g price_resident=%g profile=%s\n",
                real ? "real" : "modelled", (double) frac, (double) beta, (double) theta, (double) price_res,
                profile.empty() ? "(none)" : profile_path.c_str());
    }

    ~la_state() {
        headroom_stop = true;
        if (headroom_thread.joinable()) headroom_thread.join();
        report();
        {
            std::lock_guard<std::mutex> lock(io_mtx);
            io_stop = true;
            io_work.notify_all();
            bg_work.notify_all();
        }
        for (auto & w : io_workers) if (w.joinable()) w.join();
        for (auto & w : bg_workers) if (w.joinable()) w.join();
    }

    // LA_PROFILE_OUT: the file LA_PROFILE reads, one float per (layer, expert): how often the router asked
    // for the expert, per token. Counted from what the router wanted, not from what was served. A profile
    // given with LA_PROFILE counts as PRIOR_TOKENS tokens of earlier use, so the file follows the use of
    // the machine across runs without forgetting. Written every 512 tokens as well as at exit, because a
    // server is usually stopped by ending its process.
    void write_profile(bool say) {
        if (profile_out.empty() || layers.empty()) return;
        const double PRIOR_TOKENS = 4096.0;
        size_t n_expert = 0;
        for (auto & L : layers) if (L) n_expert = std::max(n_expert, L->asked.size());
        if (n_expert == 0) return;
        const bool prior = profile.size() == layers.size() * n_expert;
        std::vector<float> out(layers.size() * n_expert, 0.0f);
        for (size_t il = 0; il < layers.size(); ++il) {
            if (!layers[il]) continue;
            const double tokens = (double) layers[il]->clock + (prior ? PRIOR_TOKENS : 0.0);
            if (tokens <= 0) continue;
            for (size_t e = 0; e < layers[il]->asked.size(); ++e) {
                const double seen = layers[il]->asked[e] + (prior ? PRIOR_TOKENS * profile[il * n_expert + e] : 0.0);
                out[il * n_expert + e] = (float) (seen / tokens);
            }
        }
        const std::string tmp = profile_out + ".tmp";
        FILE * f = fopen(tmp.c_str(), "wb");
        if (!f) {
            if (say) fprintf(stderr, "moe-la: cannot write profile %s\n", profile_out.c_str());
            return;
        }
        const bool ok = fwrite(out.data(), sizeof(float), out.size(), f) == out.size();
        fclose(f);
        if (ok) {
            remove(profile_out.c_str());
            rename(tmp.c_str(), profile_out.c_str());
        }
        if (say) fprintf(stderr, "moe-la: profile of %zu layers x %zu experts written to %s\n", layers.size(), n_expert, profile_out.c_str());
    }

    void report() {
        if (!enabled || events == 0) return;
        const double ev = (double) events;
        const double gib = 1024.0*1024.0*1024.0;
        fprintf(stderr, "moe-la: events=%llu served/event=%.3f reads/event=%.3f exact_reads/event=%.3f "
                        "substituted/event=%.3f dropped/event=%.3f\n",
                (unsigned long long) events, served/ev, reads/ev, exact_reads/ev, substituted/ev, dropped/ev);
        if (gpu_frac > 0) fprintf(stderr, "moe-la: gpu_served/event=%.3f\n", gpu_served/ev);
        fprintf(stderr, "moe-la: %.2f s in the selection, %.2f s in splitting it between GPU and CPU\n", t_select, t_aux);
        if (gpu_adapt > 0) fprintf(stderr, "moe-la: %llu slot swaps\n", (unsigned long long) swaps);
        if (transit_runs > 0) fprintf(stderr, "moe-la: prompt batches sent %.2f GiB of experts to the GPU in %.2f s (%llu layer runs)\n",
                                      transit_bytes/gib, transit_s, (unsigned long long) transit_runs);
        if (prefetch > 0) fprintf(stderr, "moe-la: prefetch: requested/event=%.3f already_there/event=%.3f read %.2f GiB\n",
                                  pf_requested/ev, pf_already/ev, bg_bytes.load()/gib);
        if (async) fprintf(stderr, "moe-la: background: requested/event=%.3f admitted/event=%.3f skipped/event=%.3f read %.2f GiB\n",
                           bg_requested/ev, bg_admitted/ev, bg_skipped/ev, bg_bytes.load()/gib);
        if (real) {
            fprintf(stderr, "moe-la: read %.2f GiB in %.2f s of waiting (%.2f GiB/s), %llu evictions; "
                            "prefill %.2f GiB in %.2f s\n",
                    read_bytes.load()/gib, read_wait_s, read_wait_s > 0 ? read_bytes.load()/gib/read_wait_s : 0.0,
                    (unsigned long long) evicted, prefill_bytes/gib, prefill_s);
            fprintf(stderr, "moe-la: at most %d experts over the budget at one time\n", peak_over);
            fprintf(stderr, "moe-la: %.2f s in %llu page commits and releases (all threads)%s\n", g_os_ns.load() * 1e-9,
                    (unsigned long long) g_os_calls.load(), LA_WIRE ? ", pages locked (LA_WIRE)" : "");
            if (headroom > 0) fprintf(stderr, "moe-la: headroom: least available %.0f MiB; budgets lowered by %llu and raised by %llu experts\n",
                                      headroom_low == UINT64_MAX ? 0.0 : headroom_low / 1048576.0,
                                      (unsigned long long) headroom_given, (unsigned long long) headroom_taken);
            if (g_wire_fail.load() > 0) fprintf(stderr, "moe-la: LA_WIRE: %llu locks failed (ulimit -l, or the system's wired limit)\n",
                                                (unsigned long long) g_wire_fail.load());
            if (ahead_experts > 0) fprintf(stderr, "moe-la: read ahead %.2f GiB, %llu experts, %llu of them not used by the batch\n",
                                           ahead_bytes.load()/gib, (unsigned long long) ahead_experts, (unsigned long long) ahead_unused);
        }
        write_profile(true);
        if (!stats_path.empty()) {
            if (FILE * f = fopen(stats_path.c_str(), "a")) {
                fprintf(f, "{\"real\":%d,\"resident_frac\":%g,\"beta\":%g,\"wcap\":%g,\"theta\":%g,\"price_resident\":%g,"
                           "\"events\":%llu,\"served\":%llu,\"reads\":%llu,\"exact_reads\":%llu,\"substituted\":%llu,"
                           "\"dropped\":%llu,\"evicted\":%llu,\"read_bytes\":%llu,\"read_wait_s\":%.4f,"
                           "\"prefill_bytes\":%llu,\"prefill_s\":%.4f,\"async\":%d,\"bg_requested\":%llu,"
                           "\"bg_admitted\":%llu,\"bg_skipped\":%llu,\"bg_bytes\":%llu,\"gpu_served\":%llu}\n",
                        real ? 1 : 0, (double) frac, (double) beta, (double) wcap, (double) theta, (double) price_res,
                        (unsigned long long) events, (unsigned long long) served, (unsigned long long) reads,
                        (unsigned long long) exact_reads, (unsigned long long) substituted,
                        (unsigned long long) dropped, (unsigned long long) evicted,
                        (unsigned long long) read_bytes.load(), read_wait_s,
                        (unsigned long long) prefill_bytes, prefill_s, async ? 1 : 0,
                        (unsigned long long) bg_requested, (unsigned long long) bg_admitted,
                        (unsigned long long) bg_skipped, (unsigned long long) bg_bytes.load(),
                        (unsigned long long) gpu_served);
                fclose(f);
            }
        }
    }

    la_layer * fresh(int n_expert) {
        la_layer * L = new la_layer;
        L->resident.assign(n_expert, 0);
        L->last.assign(n_expert, 0);
        L->usage.assign(n_expert, 0.0f);
        L->asked.assign(n_expert, 0);
        L->bg.assign(n_expert, 0);
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

    static void expert_span(const la_tensor & t, int e, int count, uint8_t *& lo, uint8_t *& hi) {
        const uintptr_t s = (uintptr_t) t.data + (uintptr_t) e * t.stride;
        lo = (uint8_t *) page_floor(s);
        hi = (uint8_t *) std::min<uintptr_t>(page_ceil(s + (uintptr_t) count * t.stride), (uintptr_t) t.region_end);
    }

    // Queue the reads of `experts` (ascending). Neighbours in the file go into one request of at most
    // read_span bytes: the drive is asked for fewer, larger pieces. Call with io_mtx held.
    void queue_reads(la_layer & L, const std::vector<int> & experts, la_layer * owner) {
        if (io_workers.empty()) {
            for (int i = 0; i < io_threads; ++i) io_workers.emplace_back([this] { worker(); });
        }
        std::deque<la_job> & q = owner ? ahead_queue : io_queue;
        for (auto & t : L.tensors) {
            const int most = (int) std::max<size_t>(1, read_span / std::max<size_t>(1, t.stride));
            for (size_t i = 0; i < experts.size(); ) {
                size_t j = i + 1;
                while (j < experts.size() && experts[j] == experts[j - 1] + 1 && (int) (j - i) < most) ++j;
                q.push_back({ &t, experts[i], (int) (j - i), owner });
                if (owner) owner->ahead_pending++; else io_pending++;
                i = j;
            }
        }
        io_work.notify_all();
    }

    void worker() {
        std::vector<la_fd> fds;
        for (;;) {
            la_job job;
            {
                std::unique_lock<std::mutex> lock(io_mtx);
                io_work.wait(lock, [&] { return io_stop || !io_queue.empty() || !ahead_queue.empty(); });
                if (io_stop) break;
                std::deque<la_job> & q = io_queue.empty() ? ahead_queue : io_queue;
                job = q.front();
                q.pop_front();
            }
            size_t n = 0;
            const bool ok = read_one(fds, *job.t, job.expert, job.count, n);
            if (!ok) io_failed.store(true);
            (job.owner ? ahead_bytes : read_bytes).fetch_add(n);
            {
                std::lock_guard<std::mutex> lock(io_mtx);
                if (job.owner) { if (--job.owner->ahead_pending == 0) io_done.notify_all(); }
                else if (--io_pending == 0) io_done.notify_all();
            }
        }
    }

    void read_experts(la_layer & L, const std::vector<int> & experts) {
        if (L.tensors.empty() || experts.empty()) return;
        std::vector<int> sorted(experts);
        std::sort(sorted.begin(), sorted.end());
        std::unique_lock<std::mutex> lock(io_mtx);
        queue_reads(L, sorted, nullptr);
        io_done.wait(lock, [&] { return io_pending == 0; });
        if (io_failed.load()) {
            fprintf(stderr, "moe-la: reading an expert failed\n");
            abort();
        }
    }

    // Reading ahead for a batch of tokens. `want` marks the experts the next layer will probably ask
    // for (see ggml_moe_la_guess). The ones it does not hold are read while this layer computes, in
    // file order, and the layer takes them in when its own run starts.
    void read_ahead(la_layer & L, const std::vector<uint8_t> & want) {
        if (L.tensors.empty() || !L.ahead.empty()) return;
        for (int e = 0; e < (int) want.size(); ++e) {
            if (want[e] && !L.resident[e] && L.gpu_slot[e] < 0 && !L.bg[e]) L.ahead.push_back(e);
        }
        if (L.ahead.empty()) return;
        for (int e : L.ahead) L.bg[e] = 1;
        std::lock_guard<std::mutex> lock(io_mtx);
        queue_reads(L, L.ahead, &L);
    }

    // The run of `L` is about to start: wait for what was read ahead for it and take it in. Those
    // experts count as not used yet, so the ones the run does not touch are the first to go.
    bool admit_ahead(la_layer & L) {
        if (L.ahead.empty()) return false;
        {
            std::unique_lock<std::mutex> lock(io_mtx);
            io_done.wait(lock, [&] { return L.ahead_pending == 0; });
        }
        if (io_failed.load()) {
            fprintf(stderr, "moe-la: reading an expert failed\n");
            abort();
        }
        for (int e : L.ahead) {
            L.bg[e] = 0;
            if (L.resident[e]) continue;
            L.resident[e] = 1;
            L.n_resident++;
            L.last[e] = L.clock;
        }
        ahead_experts += L.ahead.size();
        return true;
    }

    bool read_one(std::vector<la_fd> & fds, const la_tensor & t, int expert, int count, size_t & n_out) {
        if ((int) fds.size() <= t.file_id) fds.resize(t.file_id + 1, LA_BAD_FD);
        if (fds[t.file_id] == LA_BAD_FD) fds[t.file_id] = os_open(files[t.file_id].c_str());
        uint8_t * lo, * hi;
        expert_span(t, expert, count, lo, hi);
        n_out = (size_t) (hi - lo);
        // lo may sit a little before data (then it holds the tail of the tensor stored before this one)
        const uint64_t off = (uint64_t) ((int64_t) t.file_off + (int64_t) (lo - t.data));
        const char * step = fds[t.file_id] == LA_BAD_FD ? "open" : !os_commit(lo, n_out) ? "commit"
                          : !os_pread(fds[t.file_id], lo, n_out, off) ? "read" : nullptr;
        if (step) {
            fprintf(stderr, "moe-la: %s failed for %zu bytes at offset %llu of %s: %s\n", step, n_out,
                    (unsigned long long) off, files[t.file_id].c_str(), strerror(errno));
        }
        return step == nullptr;
    }

    // background reader: fills the pages of an expert nobody is using yet; the layer
    // admits it the next time it starts
    void bg_worker() {
        std::vector<la_fd> fds;
        for (;;) {
            la_bg_job job;
            {
                std::unique_lock<std::mutex> lock(io_mtx);
                bg_work.wait(lock, [&] { return io_stop || !bg_queue.empty(); });
                if (io_stop) break;
                job = bg_queue.front();
                bg_queue.pop_front();
            }
            bool ok = true;
            for (auto & t : job.L->tensors) {
                size_t n = 0;
                ok = ok && read_one(fds, t, job.expert, 1, n);
                bg_bytes.fetch_add(n);
            }
            if (!ok) io_failed.store(true);
            std::lock_guard<std::mutex> lock(io_mtx);
            job.L->bg_done.push_back(job.expert);
        }
    }

    // take in the experts whose background read has finished
    void admit_bg(la_layer & L) {
        std::vector<int> done;
        if (real) {
            std::lock_guard<std::mutex> lock(io_mtx);
            done.swap(L.bg_done);
        } else {
            done.swap(L.bg_done);
        }
        for (int e : done) {
            L.bg[e] = 0;
            if (L.resident[e] || L.gpu_slot[e] >= 0) continue;
            L.resident[e] = 1;
            L.n_resident++;
            L.last[e] = L.clock + 1;                      // counts as used by the run about to start
            L.usage[e] = std::max(L.usage[e], 1.0f);
            bg_admitted++;
        }
    }

    // start background reads for the experts this run of tokens was denied, most probable first
    void request_bg(la_layer & L, int n_tokens) {
        if (L.bg_want.empty()) return;
        std::sort(L.bg_want.begin(), L.bg_want.end(), [](const std::pair<float,int> & a, const std::pair<float,int> & b) { return a.first > b.first; });
        if (real && !L.tensors.empty()) {
            std::lock_guard<std::mutex> lock(io_mtx);
            if (bg_workers.empty()) {
                for (int i = 0; i < async_threads; ++i) bg_workers.emplace_back([this] { bg_worker(); });
            }
            for (auto & w : L.bg_want) {
                if ((int) bg_queue.size() < async_queue) { bg_queue.push_back({ &L, w.second }); bg_requested++; }
                else { L.bg[w.second] = 0; bg_skipped++; }
            }
            bg_work.notify_all();
        } else {
            L.bg_credit = std::min(L.bg_credit + async_rate * n_tokens, 8.0f + async_rate * n_tokens);
            for (auto & w : L.bg_want) {
                if (L.bg_credit >= 1.0f) { L.bg_credit -= 1.0f; L.bg_done.push_back(w.second); bg_requested++; }
                else { L.bg[w.second] = 0; bg_skipped++; }
            }
        }
        L.bg_want.clear();
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
                if (L.resident[x] && !L.bg[x] && (victim < 0 || colder(L, x, victim))) victim = x;
            }
            if (victim < 0) break;
            L.resident[victim] = 0;
            L.n_resident--;
            if (real) release_expert(L, victim);
        }
    }

    static size_t expert_bytes(const la_layer & L) {
        size_t b = 0;
        for (const auto & t : L.tensors) b += t.stride;
        return b;
    }

    // LA_HEADROOM_MB: see headroom above. Budgets go down one expert per layer in turn, so every layer
    // gives up the same share; a lower budget frees memory only in a layer that is full. The layer whose
    // kernel may be running (the last one routed) is trimmed later, by ggml_moe_la_begin.
    void watch_headroom() {
        double last_say = 0;
        while (!headroom_stop) {
            std::this_thread::sleep_for(std::chrono::milliseconds(headroom_ms));
            const uint64_t avail = os_available();
            if (avail == 0) continue;
            std::lock_guard<std::mutex> lock(mtx);
            headroom_low = std::min(headroom_low, avail);
            if (layers.empty()) continue;
            if (cap0.size() < layers.size()) cap0.resize(layers.size(), -1);
            for (size_t l = 0; l < layers.size(); ++l) {
                if (layers[l] && cap0[l] < 0) cap0[l] = layers[l]->cap;
            }
            const uint64_t margin = std::max<uint64_t>(headroom / 2, (uint64_t) 512 << 20);
            int changed = 0;
            uint64_t bytes = 0;
            if (avail < headroom) {
                int64_t need = (int64_t) (headroom - avail);
                for (bool any = true; need > 0 && any; ) {
                    any = false;
                    for (size_t l = 0; l < layers.size() && need > 0; ++l) {
                        la_layer * L = layers[l].get();
                        if (!L || L->tensors.empty() || L->cap <= (int) std::ceil(headroom_floor * cap0[l])) continue;
                        L->cap--;
                        changed++;
                        any = true;
                        if (L->n_resident > L->cap) {
                            need  -= (int64_t) expert_bytes(*L);
                            bytes += expert_bytes(*L);
                        }
                    }
                }
                for (size_t l = 0; l < layers.size(); ++l) {
                    if (layers[l] && (int) l != last_end_layer) trim(*layers[l]);
                }
                headroom_given += changed;
            } else if (avail > headroom + margin) {
                // at most 256 MiB of budget a check, and never more than the room above the margin
                int64_t room = (int64_t) std::min<uint64_t>(avail - headroom - margin, (uint64_t) 256 << 20);
                for (bool any = true; room > 0 && any; ) {
                    any = false;
                    for (size_t l = 0; l < layers.size() && room > 0; ++l) {
                        la_layer * L = layers[l].get();
                        if (!L || L->tensors.empty() || L->cap >= cap0[l]) continue;
                        L->cap++;
                        changed++;
                        any = true;
                        room  -= (int64_t) expert_bytes(*L);
                        bytes += expert_bytes(*L);
                    }
                }
                headroom_taken += changed;
            }
            if (changed > 0 && now_s() - last_say > 1.0) {
                last_say = now_s();
                int cap = 0, start = 0;
                for (size_t l = 0; l < layers.size(); ++l) {
                    if (layers[l] && !layers[l]->tensors.empty()) { cap += layers[l]->cap; start += cap0[l]; }
                }
                fprintf(stderr, "moe-la: %.0f MiB available: expert budget %s by %.0f MiB, now %d of %d experts\n",
                        avail / 1048576.0, avail < headroom ? "lowered" : "raised", bytes / 1048576.0, cap, start);
            }
        }
    }

    // The slots follow the use of the model. The expert without a slot that is asked for most takes the
    // slot of the slot expert that is asked for least, when the difference is clear. In real mode the
    // two change places: the one going out is copied from the GPU into the memory the one coming in
    // gives up, so nothing is read from the disk and the memory budget does not move. Called from the
    // selection of a single token: the GPU has finished the token before and has not started this one.
    void adapt_slots(la_layer & L) {
        if (!L.slots[0] || L.n_gpu <= 0) return;
        const int n_expert = (int) L.gpu_slot.size();
        int in = -1, out = -1;
        for (int e = 0; e < n_expert; ++e) {
            if (L.gpu_slot[e] >= 0) {
                if (out < 0 || L.usage[e] < L.usage[out]) out = e;
            } else if (!L.bg[e] && (!real || L.resident[e])) {
                if (in < 0 || L.usage[e] > L.usage[in]) in = e;
            }
        }
        if (in < 0 || out < 0 || L.usage[in] < adapt_margin * L.usage[out] + 0.5f) return;
        const int slot = L.gpu_slot[out];
        for (int t = 0; t < 3; ++t) {
            if (real) {
                la_tensor & T = L.tensors[t];
                uint8_t * lo, * hi;
                expert_span(T, out, 1, lo, hi);
                os_commit(lo, (size_t) (hi - lo));
                ggml_backend_tensor_get(L.slots[t], T.data + (size_t) out * T.stride, (size_t) slot * T.stride, T.stride);
                ggml_backend_tensor_set(L.slots[t], T.data + (size_t) in * T.stride, (size_t) slot * T.stride, T.stride);
            } else {
                ggml_backend_tensor_set(L.slots[t], L.host[t] + (size_t) in * L.host_stride[t],
                                        (size_t) slot * L.host_stride[t], L.host_stride[t]);
            }
        }
        if (real) {
            release_expert(L, in);
            evicted--;                    // not an eviction
            L.resident[in]  = 0;
            L.resident[out] = 1;
            L.last[out]     = L.clock;
        }
        L.gpu_slot[in]  = (int16_t) slot;
        L.gpu_slot[out] = -1;
        swaps++;
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
    s.owned.push_back(t.data);
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
        s.last_layer = (int) (&Lp - &s.layers[0]);
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
    if (layer == 0) {
        s.pass_t0 = now_s(); s.pass_wait0 = s.read_wait_s; s.pass_bytes0 = s.read_bytes.load();
        s.pass_ahead0 = s.ahead_bytes.load(); s.pass_unused0 = s.ahead_unused; s.pass_reads0 = s.reads;
        s.pass_transit0 = s.transit_bytes; s.pass_transit_s0 = s.transit_s;
    }
    if (s.async || s.prefetch > 0) s.admit_bg(L);
    const double t0 = now_s();
    const bool ahead = s.admit_ahead(L);
    s.read_wait_s += now_s() - t0;
    if (!ahead) s.trim(L);       // what was read ahead is over the budget until the run ends
    L.run_start = L.clock + 1;
    L.pending.clear();
}

void ggml_moe_la_select(int layer, const float * probs, const float * weights, int n_expert, int k, int32_t * ids) {
    la_state & s = state();
    std::lock_guard<std::mutex> lock(s.mtx);
    la_layer & L = s.get(layer, n_expert);
    struct timer { double & acc; double t0; ~timer() { acc += now_s() - t0; } } timed{ s.t_select, now_s() };

    if (!s.real) {
        // no memory is at stake, so model generation: every token is its own run
        if (s.async || s.prefetch > 0) s.admit_bg(L);
        s.trim(L);
        L.run_start = L.clock + 1;
    }

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
    if (s.wcap > 0) {
        // Weight cap: an expert that is in neither RAM nor VRAM may be swapped for the most probable
        // expert that is, cheapest first, while the share of the top-k routing weight given up stays
        // within wcap. The share is a property of the output, so one value means the same on any model.
        // Measured with the mixing weights: with a selection bias (sigmoid routers with exp_probs_b) the
        // selection scores say little about how much of the output an expert carries.
        auto in_mem = [&](int e) { return L.resident[e] || L.gpu_slot[e] >= 0; };
        const float * wt = weights ? weights : probs;
        float wsum = 0.0f;
        for (int j = 0; j < kk; ++j) wsum += wt[wanted[j]];
        wsum = std::max(wsum, 1e-12f);
        memcpy(picked, wanted, sizeof(int) * kk);
        // candidates: experts in memory and not wanted, most probable first (idx[kk..] is unsorted)
        static thread_local std::vector<int> cand;
        cand.clear();
        for (int j = kk; j < n_expert; ++j) if (in_mem(idx[j])) cand.push_back(idx[j]);
        std::sort(cand.begin(), cand.end(), [&](int a, int b) { return probs[a] > probs[b] || (probs[a] == probs[b] && a < b); });
        // The cost of a set of swaps is how far the mix moves: the total-variation distance between the
        // wanted experts' weights and the served experts' weights, each normalized to sum 1 (the model
        // renormalizes over what it serves). It counts what is given up, what is brought in (a sigmoid
        // router's substitute may weigh more than what it replaces) and the rescaling of the rest.
        auto tv = [&](const int * set) {
            float w2 = 0.0f;
            for (int j = 0; j < kk; ++j) w2 += wt[set[j]];
            w2 = std::max(w2, 1e-12f);
            float d = 0.0f;
            for (int j = 0; j < kk; ++j) {          // wanted experts: kept (rescaled) or given up
                bool kept = false;
                for (int i = 0; i < kk; ++i) kept |= (set[i] == wanted[j]);
                const float w = wt[wanted[j]];
                d += kept ? fabsf(w / wsum - w / w2) : w / wsum;
            }
            for (int i = 0; i < kk; ++i) {          // brought in
                bool was = false;
                for (int j = 0; j < kk; ++j) was |= (set[i] == wanted[j]);
                if (!was) d += wt[set[i]] / w2;
            }
            return 0.5f * d;
        };
        // wanted and missing, cheapest by weight first; the router's first `protect` are never given up.
        // Each takes the most probable candidate whose swap keeps the distance within wcap.
        int order[64], n_order = 0;
        for (int j = std::min(s.protect, kk); j < kk; ++j) if (!in_mem(wanted[j])) order[n_order++] = j;
        std::sort(order, order + n_order, [&](int a, int b) { return wt[wanted[a]] < wt[wanted[b]]; });
        static thread_local std::vector<char> used;
        used.assign(cand.size(), 0);
        for (int o = 0; o < n_order; ++o) {
            const int j = order[o];
            if (wt[wanted[j]] / wsum > s.wcap) break;   // giving it up alone is already too much
            for (size_t c = 0; c < cand.size(); ++c) {
                if (used[c]) continue;
                const int keep = picked[j];
                picked[j] = cand[c];
                if (tv(picked) <= s.wcap) { used[c] = 1; break; }
                picked[j] = keep;
            }
        }
        std::sort(picked, picked + kk,
                  [&](int a, int b) { return probs[a] > probs[b] || (probs[a] == probs[b] && a < b); });
    } else if (s.beta > 0 || s.gamma > 0) {
        const float rng = std::max(probs[wanted[0]] - probs[wanted[kk - 1]], 1e-12f);
        adj.resize(n_expert);
        for (int e = 0; e < n_expert; ++e) {
            const float excess = L.gpu_slot[e] >= 0 ? 0.0f : L.resident[e] ? s.excess_ram : 1.0f;
            adj[e] = probs[e] * (s.gamma > 0 ? expf(-s.gamma * excess) : 1.0f) - s.beta * rng * excess;
        }
        for (int j = 0; j < std::min(s.protect, kk); ++j) adj[wanted[j]] = 2.0f + probs[wanted[j]];
        std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                          [&](int a, int b) { return adj[a] > adj[b] || (adj[a] == adj[b] && a < b); });
        for (int j = 0; j < kk; ++j) picked[j] = idx[j];
        std::sort(picked, picked + kk,
                  [&](int a, int b) { return probs[a] > probs[b] || (probs[a] == probs[b] && a < b); });
    } else {
        memcpy(picked, wanted, sizeof(int) * kk);
    }

    s.last_k = kk;
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

    if (s.async) {
        for (int j = 0; j < kk; ++j) {
            const int e = wanted[j];
            if (L.resident[e] || L.gpu_slot[e] >= 0 || L.bg[e]) continue;
            bool served_now = false;
            for (int i = 0; i < n_served; ++i) served_now |= (ids[i] == e);
            if (served_now) continue;      // it is read now, and the token waits for it
            L.bg[e] = 1;
            L.bg_want.push_back({ probs[e], e });
        }
    }

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
    for (int j = 0; j < kk; ++j) { L.usage[wanted[j]] += 1.0f; L.asked[wanted[j]]++; }
    if (!s.profile_out.empty() && layer == 0 && L.clock > 0 && (L.clock & 511) == 0) s.write_profile(false);
    if (s.decay < 1.0f && (L.clock & 15) == 0) {
        for (float & u : L.usage) u *= s.decay;
    }
    for (int j = 0; j < n_served; ++j) L.last[ids[j]] = L.clock;
    for (int j = 0; j < n_served; ++j) {
        const int e = ids[j];
        if (L.gpu_slot[e] >= 0) { s.gpu_served++; continue; }
        if (L.resident[e]) continue;
        s.reads++;
        if (!s.real && L.n_resident >= L.cap) {
            int victim = -1;
            for (int x = 0; x < n_expert; ++x) {
                if (!L.resident[x] || L.bg[x] || L.last[x] >= L.run_start) continue;
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
    if (!s.real && s.async) s.request_bg(L, 1);
}

void ggml_moe_la_end(int layer) {
    la_state & s = state();
    std::lock_guard<std::mutex> lock(s.mtx);
    if ((int) s.layers.size() <= layer || !s.layers[layer]) return;
    la_layer & L = *s.layers[layer];
    s.last_end_layer = layer;
    const int n_tokens = (int) (L.clock + 1 - L.run_start);
    if (s.real) {
        // Room for what the run reads is made here, when everything the run uses is known: an expert
        // evicted for the first token of a batch could be one that a later token needs, and it
        // would be read twice. Only experts the run does not use can go, the coldest first.
        const int n_expert = (int) L.resident.size();
        for (int e : L.ahead) if (L.last[e] < L.run_start) s.ahead_unused++;
        L.ahead.clear();
        while (L.n_resident > L.cap) {
            int victim = -1;
            for (int x = 0; x < n_expert; ++x) {
                if (!L.resident[x] || L.bg[x] || L.last[x] >= L.run_start) continue;
                if (victim < 0 || s.colder(L, x, victim)) victim = x;
            }
            if (victim < 0) break;
            L.resident[victim] = 0;
            L.n_resident--;
            s.release_expert(L, victim);
        }
        if (!L.pending.empty()) {
            const double t0 = now_s();
            s.read_experts(L, L.pending);
            s.read_wait_s += now_s() - t0;
        }
        if (s.gpu_adapt > 0 && n_tokens == 1 && L.clock % (uint64_t) s.gpu_adapt == 0 && (int) L.tensors.size() == 3) s.adapt_slots(L);
        if (layer == s.last_layer && n_tokens >= 32) {
            const double gib = 1024.0*1024.0*1024.0;
            fprintf(stderr, "moe-la: batch of %d tokens: %.1f s, of which %.1f s waiting for the disk; read %.2f GiB after routing "
                            "(%.1f experts/layer) and %.2f GiB ahead (%llu experts not used)\n",
                    n_tokens, now_s() - s.pass_t0, s.read_wait_s - s.pass_wait0, (s.read_bytes.load() - s.pass_bytes0)/gib,
                    (double) (s.reads - s.pass_reads0) / (s.last_layer + 1), (s.ahead_bytes.load() - s.pass_ahead0)/gib,
                    (unsigned long long) (s.ahead_unused - s.pass_unused0));
            if (s.transit_bytes > s.pass_transit0) {
                // counts the layers before this one: the last layer sends its experts after this line
                fprintf(stderr, "moe-la:   %.2f GiB of experts sent to the GPU in %.2f s\n",
                        (s.transit_bytes - s.pass_transit0)/gib, s.transit_s - s.pass_transit_s0);
            }
        }
    }
    if (!s.real && s.gpu_adapt > 0 && n_tokens == 1 && L.clock % (uint64_t) s.gpu_adapt == 0 && L.host[0]) s.adapt_slots(L);
    L.pending.clear();
    if (s.async) s.request_bg(L, n_tokens);
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
    struct timer { double & acc; double t0; ~timer() { acc += now_s() - t0; } } timed{ s.t_aux, now_s() };
    // kind 1: a selected expert that runs on the CPU still needs a slot id on the GPU side. It gets a
    // slot that no expert of this token uses, a different one each time, and the caller zeroes its
    // output with the kind 3 mask. The ids of one token must stay distinct: CUDA's kernel groups
    // tokens by expert and counts each expert at most once per token.
    const int n_gpu = L ? L->n_gpu : 0;
    int spare = 0;
    auto taken = [&](int c) {
        for (int i = 0; i < k; ++i) {
            const int e = ids[i];
            if (L && e >= 0 && e < (int) L->gpu_slot.size() && L->gpu_slot[e] == c) return true;
        }
        return false;
    };
    for (int j = 0; j < k; ++j) {
        const int e = ids[j];
        const int slot = (L && e >= 0 && e < (int) L->gpu_slot.size()) ? L->gpu_slot[e] : -1;
        switch (kind) {
            case 1:
                if (slot >= 0) { ((int32_t *) out)[j] = slot; break; }
                while (spare < n_gpu && taken(spare)) ++spare;
                ((int32_t *) out)[j] = spare < n_gpu ? spare++ : 0;   // a layer has at least k slots, see llama.cpp
                break;
            case 2: ((int32_t *) out)[j] = slot >= 0 ? -1 : e;   break;
            case 6: ((float *) out)[j]   = (e >= 0 && slot < 0) ? 1.0f : 0.0f; break;
            default: ((float *) out)[j]  = slot >= 0 ? 1.0f : 0.0f; break;
        }
    }
}

int ggml_moe_la_gpu_batch_min(void) {
    const la_state & s = state();
    return s.enabled && s.gpu_frac > 0 && s.real && s.gpu_batch_given ? s.gpu_batch : 0;
}

bool ggml_moe_la_batch_plain(int n_tokens) {
    const la_state & s = state();
    // a dropped expert is the id -1, which only the CPU kernels understand
    return s.enabled && s.gpu_frac > 0 && !s.real && s.theta <= 0 && s.gpu_batch > 0 && n_tokens >= s.gpu_batch;
}

void ggml_moe_la_set_slots(int layer, struct ggml_tensor * const slots[3], const void * const host[3], const size_t stride[3]) {
    la_state & s = state();
    std::lock_guard<std::mutex> lock(s.mtx);
    if ((int) s.layers.size() <= layer || !s.layers[layer]) return;
    la_layer & L = *s.layers[layer];
    for (int t = 0; t < 3; ++t) {
        L.slots[t] = slots[t];
        L.host[t]  = (const uint8_t *) host[t];
        L.host_stride[t] = stride[t];
    }
}

void ggml_moe_la_set_transit(int layer, struct ggml_tensor * gate, struct ggml_tensor * up, struct ggml_tensor * down) {
    la_state & s = state();
    std::lock_guard<std::mutex> lock(s.mtx);
    if ((int) s.layers.size() <= layer || !s.layers[layer]) return;
    la_layer & L = *s.layers[layer];
    L.transit[0] = gate; L.transit[1] = up; L.transit[2] = down;
}

// A prompt batch on a machine with a GPU. The experts of the batch that have no slot are in host
// memory by now (the selection of this layer has run). They are copied into the scratch tensors of
// the layer, at their own index, and the whole layer then runs on the GPU. out: for each selected
// entry the index to use there. An entry that has a slot (or was dropped) gets an index its token
// does not use; the caller zeroes its output with the kind 6 mask. Whatever the scratch tensors
// hold at such an index is finite: zeros from the start, or an expert of an earlier layer.
void ggml_moe_la_transit(int layer, const int32_t * ids, int k, int n_tokens, size_t ids_stride,
                         const void * const src[3], const size_t stride[3], int32_t * out, size_t out_stride) {
    la_state & s = state();
    std::lock_guard<std::mutex> lock(s.mtx);
    la_layer & L = *s.layers[layer];
    const int n_expert = (int) L.gpu_slot.size();
    std::vector<uint8_t> need(n_expert, 0);
    for (int i = 0; i < n_tokens; ++i) {
        const int32_t * row = (const int32_t *) ((const char *) ids + (size_t) i * ids_stride);
        int32_t * o = (int32_t *) ((char *) out + (size_t) i * out_stride);
        int spare = 0;
        auto taken = [&](int c) { for (int j = 0; j < k; ++j) if (row[j] == c) return true; return false; };
        for (int j = 0; j < k; ++j) {
            const int e = row[j];
            if (e >= 0 && e < n_expert && L.gpu_slot[e] < 0) { need[e] = 1; o[j] = e; continue; }
            while (spare < n_expert && taken(spare)) ++spare;
            o[j] = spare < n_expert ? spare++ : 0;
        }
    }
    const double t0 = now_s();
    for (int t = 0; t < 3; ++t) {
        for (int e = 0; e < n_expert; ) {
            if (!need[e]) { ++e; continue; }
            int j = e + 1;
            while (j < n_expert && need[j]) ++j;
            ggml_backend_tensor_set(L.transit[t], (const char *) src[t] + (size_t) e * stride[t],
                                    (size_t) e * stride[t], (size_t) (j - e) * stride[t]);
            s.transit_bytes += (uint64_t) (j - e) * stride[t];
            e = j;
        }
    }
    s.transit_s += now_s() - t0;
    s.transit_runs++;
}

bool ggml_moe_la_prefetching(int n_tokens) {
    const la_state & s = state();
    return s.enabled && (s.prefetch > 0 || (s.real && s.ahead_k != 0 && n_tokens > 1));
}

// The same guess for a whole batch: the union over its tokens is read ahead. A long prompt batch needs
// nearly every expert of a layer, a short one a part of them that the router of the layer names well
// enough even from the input of the layer before.
void ggml_moe_la_guess(int layer, const float * logits, int n_expert, int n_rows, size_t row_bytes) {
    la_state & s = state();
    if (!s.enabled) return;
    if (n_rows < 2) {
        if (n_rows == 1) ggml_moe_la_prefetch(layer, logits, n_expert);
        return;
    }
    if (!s.real || s.ahead_k == 0) return;
    std::lock_guard<std::mutex> lock(s.mtx);
    la_layer & L = s.get(layer, n_expert);
    const int m = std::min(n_expert, s.ahead_k > 0 ? s.ahead_k : std::max(1, s.last_k));
    static thread_local std::vector<int> idx;
    std::vector<uint8_t> want(n_expert, 0);
    for (int r = 0; r < n_rows; ++r) {
        const float * row = (const float *) ((const char *) logits + (size_t) r * row_bytes);
        idx.resize(n_expert);
        std::iota(idx.begin(), idx.end(), 0);
        if (m < n_expert) std::nth_element(idx.begin(), idx.begin() + m, idx.end(), [&](int a, int b) { return row[a] > row[b]; });
        for (int j = 0; j < m; ++j) want[idx[j]] = 1;
    }
    s.read_ahead(L, want);
}

// The router of `layer` applied to the input of the layer before it: a guess at what
// `layer` will ask for. Start reading the best guesses that are not in memory, so the
// read overlaps the computation of the layer before.
void ggml_moe_la_prefetch(int layer, const float * logits, int n_expert) {
    la_state & s = state();
    if (!s.enabled || s.prefetch <= 0) return;
    std::lock_guard<std::mutex> lock(s.mtx);
    la_layer & L = s.get(layer, n_expert);
    static thread_local std::vector<int> idx;
    idx.resize(n_expert);
    std::iota(idx.begin(), idx.end(), 0);
    const int m = std::min(s.prefetch, n_expert);
    std::partial_sort(idx.begin(), idx.begin() + m, idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    for (int j = 0; j < m; ++j) {
        const int e = idx[j];
        if (L.resident[e] || L.gpu_slot[e] >= 0) { s.pf_already++; continue; }
        if (L.bg[e]) continue;
        L.bg[e] = 1;
        s.pf_requested++;
        if (s.real && !L.tensors.empty()) {
            std::lock_guard<std::mutex> io(s.io_mtx);
            if (s.bg_workers.empty()) {
                for (int i = 0; i < s.async_threads; ++i) s.bg_workers.emplace_back([&s] { s.bg_worker(); });
            }
            s.bg_queue.push_front({ &L, e });     // ahead of plain background loads: it is needed within this token
            s.bg_work.notify_all();
        } else {
            L.bg_done.push_back(e);               // modelled: there by the time the layer starts
        }
    }
}

// True for a tensor whose memory the cache owns: only the resident experts are backed, so
// nothing may read it as a whole (for example to copy it to a GPU).
bool ggml_moe_la_owns(const void * data) {
    la_state & s = state();
    if (!s.enabled || !s.real || !data) return false;
    return std::find(s.owned.begin(), s.owned.end(), data) != s.owned.end();
}
