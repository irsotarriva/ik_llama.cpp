// Included at the end of ggml-moe-la.cpp: it uses the selector's state.
//
// ======================================================================================================
// VRAM tier, fast path (LA_FAST, on by default where it applies)
//
// The tier as above makes about a dozen GPU <-> CPU round trips per layer while generating: the CPU
// picks the experts, splits the ids for the two sides, and the two halves of the experts run one after
// the other. When the residency cannot change, none of that needs the CPU: the selection and the slot
// ids are computed on the GPU (ggml-cuda/moe-la-fast.cu), and the experts without a slot are computed
// here by a worker thread. A GPU kernel writes the layer's input, the selection and the weights to a
// mailbox in mapped host memory and raises a flag; the GPU then runs the slot experts while the worker
// computes the others; a second kernel waits for the worker's flag and adds its result. One round trip
// per layer, both halves at once, and the whole token stays one CUDA graph.
// ======================================================================================================

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace {

struct la_fast_layer {
    struct ggml_tensor * t[3] = { nullptr, nullptr, nullptr };   // gate, up, down in host memory
    float swiglu_limit = 0.0f;
    bool  fused = true;
    // the worker's graph for each batch size (index = n_tokens), in the context of that batch size
    struct graph {
        struct ggml_cgraph  * gf  = nullptr;
        struct ggml_tensor  * x = nullptr, * ids = nullptr, * w = nullptr, * out = nullptr;
    };
    std::vector<graph> graphs;
};

struct la_fast_state {
    bool   read = false;
    bool   on = true;          // LA_FAST
    int    max_tokens = 8;     // LA_FAST_MAX
    int    n_threads = 8;      // LA_FAST_THREADS
    bool   static_ok = true;   // every layer: each expert in a slot or resident, and nothing can move it
    std::vector<la_fast_layer> layers;
    // One ggml context per batch size holds the graphs of every layer: ggml allows only
    // GGML_MAX_CONTEXTS (64) at once, so a context per layer and size ran out at the second size.
    std::vector<struct ggml_context *> size_ctx;
    std::mutex mtx;            // registration and graph building
    std::vector<uint8_t> work;
    // the worker
    ggml_moe_la_mailbox * mb = nullptr;
    size_t off_x = 0, off_ids = 0, off_w = 0, off_out = 0;
    std::thread worker;
    std::atomic<bool> stop{false};
    uint64_t jobs = 0, cpu_jobs = 0, cpu_experts = 0, skipped = 0;
    double busy_s = 0.0, compute_s = 0.0;
    int report_every = 0;      // LA_FAST_REPORT
    double rep_busy = 0.0, rep_comp = 0.0;
    uint64_t rep_cpu_jobs = 0;

    void read_env() {
        if (read) return;
        read = true;
        if (const char * v = getenv("LA_FAST"))         on = atoi(v) != 0;
        if (const char * v = getenv("LA_FAST_MAX"))     max_tokens = std::max(1, atoi(v));
        const unsigned hw = std::thread::hardware_concurrency();
        n_threads = hw >= 16 ? 8 : (int) std::max(1u, hw / 2);
        if (const char * v = getenv("LA_FAST_THREADS")) n_threads = std::max(1, atoi(v));
        if (const char * v = getenv("LA_FAST_REPORT"))  report_every = std::max(0, atoi(v));
    }
    ~la_fast_state() {
        stop = true;
        if (worker.joinable()) worker.join();
        if (jobs > 0) {
            fprintf(stderr, "moe-la: fast path: %llu worker jobs, %.1f us each; %llu with experts on the CPU (%.2f experts, %.1f us of compute each)\n",
                    (unsigned long long) jobs, 1e6 * busy_s / (double) jobs, (unsigned long long) cpu_jobs,
                    cpu_jobs ? (double) cpu_experts / (double) cpu_jobs : 0.0, cpu_jobs ? 1e6 * compute_s / (double) cpu_jobs : 0.0);
        }
        if (mb) {
            fprintf(stderr, "moe-la: fast path: last job %d started, %d done, error flag %d, %llu jobs seen late\n",
                    (int) mb->start_seq, (int) mb->done_seq, (int) mb->error, (unsigned long long) skipped);
        }
        for (auto * c : size_ctx) if (c) ggml_free(c);
    }
};

la_fast_state & fast() {
    static la_fast_state f;
    return f;
}

// The worker's graph for one layer and batch size: the experts without a slot, weighted and summed.
// ids of entries that run in a slot are -1, which the CPU kernels skip (zero output).
la_fast_layer::graph & fast_graph(la_fast_layer & L, int n_tokens, int k, int n_embd) {
    la_fast_state & f = fast();
    if ((int) L.graphs.size() <= n_tokens) L.graphs.resize(n_tokens + 1);
    auto & g = L.graphs[n_tokens];
    if (g.gf) return g;
    if ((int) f.size_ctx.size() <= n_tokens) f.size_ctx.resize(n_tokens + 1, nullptr);
    if (!f.size_ctx[n_tokens]) {
        // room for the graph of every registered layer at this size
        size_t mem = 64 * 1024;
        for (const auto & FL : f.layers) {
            if (!FL.t[0]) continue;
            const size_t n_ff = (size_t) FL.t[0]->ne[1];
            const size_t floats = (size_t) n_tokens * (2 * (size_t) n_embd + 2 * (size_t) k + 3 * (size_t) k * n_ff + (size_t) k * n_embd);
            mem += ggml_tensor_overhead() * 32 + ggml_graph_overhead() + sizeof(float) * floats + 4096;
        }
        struct ggml_init_params ip = { mem, nullptr, false };
        f.size_ctx[n_tokens] = ggml_init(ip);
        GGML_ASSERT(f.size_ctx[n_tokens] && "moe-la fast path: no ggml context left");
    }
    struct ggml_context * ctx = f.size_ctx[n_tokens];
    g.x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
    g.ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, n_tokens);
    g.w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, k, n_tokens);
    struct ggml_tensor * par;
    if (L.fused && L.t[0]->type == L.t[1]->type) {
        par = ggml_moe_up_gate(ctx, L.t[1], L.t[0], g.x, g.ids, GGML_UNARY_OP_SILU);
    } else {
        struct ggml_tensor * gate = ggml_mul_mat_id(ctx, L.t[0], g.x, g.ids);
        struct ggml_tensor * up   = ggml_mul_mat_id(ctx, L.t[1], g.x, g.ids);
        par = ggml_fused_mul_unary(ctx, gate, up, GGML_UNARY_OP_SILU);
    }
    *((float *)(par->op_params + 1)) = L.swiglu_limit;
    struct ggml_tensor * ex = ggml_mul_mat_id(ctx, L.t[2], par, g.ids);
    g.out = ggml_mul_multi_add(ctx, ex, g.w);
    g.gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(g.gf, g.out);
    return g;
}

// The experts of a job that have no slot. ids: the selection (k per token); w: the weights.
void fast_run(int layer, int n_tokens, int k, int n_embd, const float * x, const int32_t * ids,
              const float * w, float * out) {
    la_state & s = state();
    la_fast_state & f = fast();
    const double t0 = now_s();
    la_fast_layer & FL = f.layers[layer];
    la_layer & L = *s.layers[layer];
    la_fast_layer::graph * g;
    {
        std::lock_guard<std::mutex> lock(f.mtx);
        g = &fast_graph(FL, n_tokens, k, n_embd);
    }
    memcpy(g->x->data, x, sizeof(float) * (size_t) n_embd * n_tokens);
    memcpy(g->w->data, w, sizeof(float) * (size_t) k * n_tokens);
    int32_t * cid = (int32_t *) g->ids->data;
    int n_cpu = 0, n_gpu = 0;
    for (int i = 0; i < k * n_tokens; ++i) {
        const int e = ids[i];
        const bool slot = e >= 0 && e < (int) L.gpu_slot.size() && L.gpu_slot[e] >= 0;
        cid[i] = slot ? -1 : e;
        n_cpu += !slot; n_gpu += slot;
    }
    if (n_cpu > 0) {
        const double c0 = now_s();
        struct ggml_cplan plan = ggml_graph_plan(g->gf, f.n_threads);
        if (plan.work_size > f.work.size()) f.work.resize(plan.work_size);
        plan.work_data = f.work.data();
        ggml_graph_compute(g->gf, &plan);
        memcpy(out, g->out->data, sizeof(float) * (size_t) n_embd * n_tokens);
        f.compute_s += now_s() - c0;
        f.cpu_jobs++;
        f.cpu_experts += n_cpu;
    } else {
        memset(out, 0, sizeof(float) * (size_t) n_embd * n_tokens);
    }
    // the counters the CPU selection would have kept
    {
        std::lock_guard<std::mutex> lock(s.mtx);
        s.events += n_tokens;
        s.served += (uint64_t) k * n_tokens;
        s.gpu_served += n_gpu;
        for (int i = 0; i < k * n_tokens; ++i) {
            const int e = ids[i];
            if (e < 0 || e >= (int) L.usage.size()) continue;
            L.usage[e] += 1.0f; L.asked[e]++;
        }
        L.clock += n_tokens;
        if (!s.profile_out.empty() && layer == 0 && (L.clock & 511) < (uint64_t) n_tokens) s.write_profile(false);
    }
    f.jobs++;
    f.busy_s += now_s() - t0;
    if (f.report_every > 0 && f.jobs % (uint64_t) f.report_every == 0) {
        // LA_FAST_REPORT=N: every N jobs, the averages since the last report (a server is not stopped cleanly)
        const double busy = f.busy_s - f.rep_busy, comp = f.compute_s - f.rep_comp;
        const uint64_t cj = f.cpu_jobs - f.rep_cpu_jobs;
        fprintf(stderr, "moe-la: fast path, last %d jobs: %.1f us each, %llu with CPU experts at %.1f us of compute\n",
                f.report_every, 1e6 * busy / f.report_every, (unsigned long long) cj, cj ? 1e6 * comp / (double) cj : 0.0);
        f.rep_busy = f.busy_s; f.rep_comp = f.compute_s; f.rep_cpu_jobs = f.cpu_jobs;
    }
}

void fast_worker_loop() {
    la_fast_state & f = fast();
    ggml_moe_la_mailbox * mb = f.mb;
    int32_t seen = mb->start_seq;
    char * base = (char *) mb;
    int idle = 0;
    while (!f.stop) {
        const int32_t seq = mb->start_seq;
        if (seq == seen) {
            // spin while jobs come in quickly (a token is ~40 jobs a few hundred microseconds apart),
            // then back off so an idle server does not burn a core
            if (++idle < 200000) {
#if defined(_M_X64) || defined(__x86_64__)
                _mm_pause();
#endif
            } else if (idle < 400000) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
            continue;
        }
        idle = 0;
        if (seq != seen + 1) f.skipped += (uint64_t) (seq - seen - 1);   // jobs the GPU gave up waiting for
        seen = seq;
        std::atomic_thread_fence(std::memory_order_acquire);
        const int layer = mb->layer, n_tokens = mb->n_tokens, k = mb->k, n_embd = mb->n_embd;
        if (layer < 0 || layer >= (int) f.layers.size() || !f.layers[layer].t[0]) {
            mb->error = 1;
        } else {
            fast_run(layer, n_tokens, k, n_embd, (const float *) (base + f.off_x), (const int32_t *) (base + f.off_ids),
                     (const float *) (base + f.off_w), (float *) (base + f.off_out));
        }
        std::atomic_thread_fence(std::memory_order_release);
        mb->done_seq = seq;
    }
}

} // namespace

bool ggml_moe_la_fast_ok(int n_tokens) {
    la_state & s = state();
    la_fast_state & f = fast();
    f.read_env();
    return f.on && f.static_ok && s.enabled && s.gpu_frac > 0 && !s.real && s.gpu_adapt == 0 && s.wcap <= 0 &&
           s.theta <= 0 && !s.async && s.prefetch <= 0 && n_tokens >= 1 && n_tokens <= f.max_tokens;
}

int ggml_moe_la_fast_max_tokens(void) {
    la_fast_state & f = fast();
    f.read_env();
    return f.max_tokens;
}

void ggml_moe_la_fast_register(int layer, struct ggml_tensor * gate, struct ggml_tensor * up,
                               struct ggml_tensor * down, float swiglu_limit, bool fused_up_gate) {
    la_state & s = state();
    la_fast_state & f = fast();
    f.read_env();
    // The engine's own CPU threads have almost nothing to do while the fast path generates, but by default
    // they spin between tokens and take the cores from the worker: in the server that cut generation to a
    // quarter. Unless the user chose otherwise, they wait passively (read by OpenMP at its first parallel
    // region, which comes after the model is loaded).
    if (f.on && !getenv("OMP_WAIT_POLICY")) {
#ifdef _WIN32
        _putenv_s("OMP_WAIT_POLICY", "passive");
#else
        setenv("OMP_WAIT_POLICY", "passive", 0);
#endif
    }
    std::lock_guard<std::mutex> lock(f.mtx);
    if ((int) f.layers.size() <= layer) f.layers.resize(layer + 1);
    auto & L = f.layers[layer];
    L.t[0] = gate; L.t[1] = up; L.t[2] = down;
    L.swiglu_limit = swiglu_limit;
    L.fused = fused_up_gate;
    // the GPU tables are made once: every expert must stay where it is
    std::lock_guard<std::mutex> lock2(s.mtx);
    if ((int) s.layers.size() > layer && s.layers[layer]) {
        const la_layer & SL = *s.layers[layer];
        for (size_t e = 0; e < SL.gpu_slot.size(); ++e) {
            if (SL.gpu_slot[e] < 0 && !SL.resident[e]) { f.static_ok = false; break; }
        }
    } else {
        f.static_ok = false;
    }
}

int ggml_moe_la_fast_tables(int layer, int n_expert, int32_t * slot, float * excess) {
    la_state & s = state();
    std::lock_guard<std::mutex> lock(s.mtx);
    la_layer & L = s.get(layer, n_expert);
    for (int e = 0; e < n_expert; ++e) {
        slot[e]   = L.gpu_slot[e];
        excess[e] = L.gpu_slot[e] >= 0 ? 0.0f : L.resident[e] ? s.excess_ram : 1.0f;
    }
    return L.n_gpu;
}

void ggml_moe_la_fast_params(float * beta, float * gamma, int * protect) {
    const la_state & s = state();
    *beta = s.beta; *gamma = s.gamma; *protect = s.protect;
}

// Must give the same experts as ggml_moe_la_select for the same tables (theta 0, no weight cap).
void ggml_moe_la_fast_pick(const float * probs, int n_expert, int k, const int32_t * slot, const float * excess,
                           float beta, float gamma, int protect, int32_t * ids) {
    (void) slot;
    static thread_local std::vector<int> idx;
    static thread_local std::vector<float> adj;
    idx.resize(n_expert);
    std::iota(idx.begin(), idx.end(), 0);
    const int kk = std::min(k, 64);
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                      [&](int a, int b) { return probs[a] > probs[b] || (probs[a] == probs[b] && a < b); });
    int wanted[64];
    for (int j = 0; j < kk; ++j) wanted[j] = idx[j];
    if (beta > 0 || gamma > 0) {
        const float rng = std::max(probs[wanted[0]] - probs[wanted[kk - 1]], 1e-12f);
        adj.resize(n_expert);
        for (int e = 0; e < n_expert; ++e) {
            adj[e] = probs[e] * (gamma > 0 ? expf(-gamma * excess[e]) : 1.0f) - beta * rng * excess[e];
        }
        for (int j = 0; j < std::min(protect, kk); ++j) adj[wanted[j]] = 2.0f + probs[wanted[j]];
        std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                          [&](int a, int b) { return adj[a] > adj[b] || (adj[a] == adj[b] && a < b); });
        for (int j = 0; j < kk; ++j) ids[j] = idx[j];
        std::sort(ids, ids + kk, [&](int a, int b) { return probs[a] > probs[b] || (probs[a] == probs[b] && a < b); });
    } else {
        for (int j = 0; j < kk; ++j) ids[j] = wanted[j];
    }
}

size_t ggml_moe_la_mailbox_layout(int n_embd, int k, int max_tokens, size_t * off_x, size_t * off_ids,
                                  size_t * off_w, size_t * off_out) {
    auto up = [](size_t v) { return (v + 255) & ~(size_t) 255; };
    size_t o = up(sizeof(ggml_moe_la_mailbox));
    *off_x   = o; o = up(o + sizeof(float)   * (size_t) n_embd * max_tokens);
    *off_ids = o; o = up(o + sizeof(int32_t) * (size_t) k * max_tokens);
    *off_w   = o; o = up(o + sizeof(float)   * (size_t) k * max_tokens);
    *off_out = o; o = up(o + sizeof(float)   * (size_t) n_embd * max_tokens);
    return o;
}

#ifdef _WIN32
#include <psapi.h>
static DWORD g_la_worker_tid = 0;
// LA_FAST_CRASH=1: print where an access violation happened (module and offset, and whether it was the worker)
static LONG WINAPI la_fast_crash(EXCEPTION_POINTERS * ep) {
    const DWORD code = ep->ExceptionRecord->ExceptionCode;
    if (code != EXCEPTION_ACCESS_VIOLATION && code != EXCEPTION_ILLEGAL_INSTRUCTION) return EXCEPTION_CONTINUE_SEARCH;
    void * addr = ep->ExceptionRecord->ExceptionAddress;
    HMODULE mod = nullptr;
    char name[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR) addr, &mod)) {
        GetModuleFileNameA(mod, name, MAX_PATH);
    }
    fprintf(stderr, "moe-la: crash 0x%08lx at %s+0x%llx (%s thread), reading/writing %p\n", code, name,
            (unsigned long long) ((char *) addr - (char *) mod), GetCurrentThreadId() == g_la_worker_tid ? "worker" : "other",
            ep->ExceptionRecord->NumberParameters > 1 ? (void *) ep->ExceptionRecord->ExceptionInformation[1] : nullptr);
    fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

void ggml_moe_la_fast_attach(struct ggml_moe_la_mailbox * mb, int n_embd, int k, int max_tokens) {
    la_fast_state & f = fast();
    if (f.worker.joinable()) return;
#ifdef _WIN32
    if (const char * v = getenv("LA_FAST_CRASH"); v && atoi(v)) AddVectoredExceptionHandler(1, la_fast_crash);
#endif
    f.mb = mb;
    ggml_moe_la_mailbox_layout(n_embd, k, max_tokens, &f.off_x, &f.off_ids, &f.off_w, &f.off_out);
    fprintf(stderr, "moe-la: fast path on (selection on the GPU, %d worker threads, batches up to %d tokens)\n",
            f.n_threads, f.max_tokens);
    f.worker = std::thread(fast_worker_loop);
#ifdef _WIN32
    g_la_worker_tid = GetThreadId((HANDLE) f.worker.native_handle());
#endif
}

void ggml_moe_la_fast_compute(int layer, int n_tokens, int k, int n_embd, const float * x, const int32_t * ids,
                              const float * w, float * out) {
    fast_run(layer, n_tokens, k, n_embd, x, ids, w, out);
}
