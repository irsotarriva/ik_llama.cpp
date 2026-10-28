// VRAM tier, fast path of the load-aware selector.
//
// While generating, a layer of the tier used to go GPU (router) -> CPU (selection and three helper
// ops) -> GPU (slot experts) and CPU (the other experts, after the GPU) -> GPU, every hand-over a
// synchronous copy. Here the selection and the slot ids are kernels, and the experts without a slot
// are handed to a worker thread through a mailbox in mapped host memory:
//
//   select (-20)  probs -> the experts, the same rule as ggml_moe_la_select (bias, protect, excess)
//   map (-21/-23) the slot id of each entry, and the mask of the entries that run in a slot
//   start (-24)   writes the layer's input, the selection and the weights to the mailbox, raises a flag
//   ...           the slot experts run on the GPU while the worker computes the others
//   collect (-25) waits for the worker's flag and adds its result
//
// Everything stays in one stream and is captured in the token's CUDA graph; the job number comes from
// a counter in device memory, so a replayed graph numbers its jobs as a new one would.

#include "moe-la-fast.cuh"
#include "../ggml-moe-la.h"
#include "../../include/ggml-cuda.h"

#include <cstdio>
#include <vector>

namespace {

struct la_fast_dev {
    int device = -1;
    int n_embd = 0, k = 0, max_tokens = 0, n_expert = 0;
    ggml_moe_la_mailbox * mb_host = nullptr;
    char * mb_dev = nullptr;
    size_t off_x = 0, off_ids = 0, off_w = 0, off_out = 0;
    int * seq_dev = nullptr;
    std::vector<int32_t *> slot;
    std::vector<float *>   excess;
    std::vector<int>       n_gpu;
    float beta = 0.0f, gamma = 0.0f;
    int protect = 0;
    int skip = 1;      // LA_FAST_SKIP: a job whose experts are all in slots is not sent to the worker
};

la_fast_dev g_fast;

constexpr int LA_MAX_K = 64;
constexpr int LA_MAX_EXPERT = 1024;

// k times: the largest value not taken yet, the lower index on a tie (as std::partial_sort with the
// selector's comparator). One warp; v and taken in shared memory.
__device__ void la_topk_warp(const float * v, char * taken, int n, int k, int * out) {
    const int lane = threadIdx.x;
    for (int e = lane; e < n; e += WARP_SIZE) taken[e] = 0;
    __syncwarp();
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY;
        int   bi = INT_MAX;
        for (int e = lane; e < n; e += WARP_SIZE) {
            if (taken[e]) continue;
            const float x = v[e];
            if (x > bv || (x == bv && e < bi)) { bv = x; bi = e; }
        }
        for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, bv, off);
            const int   oi = __shfl_xor_sync(0xffffffff, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) {
            out[r] = bi;
            if (bi >= 0 && bi < n) taken[bi] = 1;
        }
        __syncwarp();
    }
}

__global__ void k_la_select(const float * probs, int64_t row_stride, int n_expert, int k, int32_t * ids, int64_t ids_stride,
                            const float * excess, float beta, float gamma, int protect) {
    __shared__ float adj[LA_MAX_EXPERT];
    __shared__ char  taken[LA_MAX_EXPERT];
    __shared__ int   wanted[LA_MAX_K];
    __shared__ int   picked[LA_MAX_K];
    const float * p = probs + blockIdx.x * row_stride;
    const int lane = threadIdx.x;
    la_topk_warp(p, taken, n_expert, k, wanted);
    if (beta > 0.0f || gamma > 0.0f) {
        const float rng = fmaxf(p[wanted[0]] - p[wanted[k - 1]], 1e-12f);
        for (int e = lane; e < n_expert; e += WARP_SIZE) {
            adj[e] = p[e] * (gamma > 0.0f ? expf(-gamma * excess[e]) : 1.0f) - beta * rng * excess[e];
        }
        __syncwarp();
        if (lane < min(protect, k)) adj[wanted[lane]] = 2.0f + p[wanted[lane]];
        __syncwarp();
        la_topk_warp(adj, taken, n_expert, k, picked);
        if (lane == 0) {
            // most probable first, as the CPU selection returns them
            for (int i = 1; i < k; ++i) {
                const int x = picked[i];
                int j = i - 1;
                while (j >= 0 && (p[picked[j]] < p[x] || (p[picked[j]] == p[x] && picked[j] > x))) { picked[j + 1] = picked[j]; --j; }
                picked[j + 1] = x;
            }
        }
        __syncwarp();
    } else {
        if (lane < k) picked[lane] = wanted[lane];
        __syncwarp();
    }
    if (lane < k) ids[blockIdx.x * ids_stride + lane] = picked[lane];
}

// as ggml_moe_la_aux kinds 1 and 3
__global__ void k_la_map(const int32_t * ids, int64_t ids_stride, int k, int n_tokens, int n_expert, const int32_t * slot,
                         int n_gpu, int kind, char * out, int64_t out_stride) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tokens) return;
    const int32_t * row = ids + t * ids_stride;
    auto slot_of = [&](int e) { return (e >= 0 && e < n_expert) ? slot[e] : -1; };
    int spare = 0;
    for (int j = 0; j < k; ++j) {
        const int s = slot_of(row[j]);
        if (kind == 3) {
            ((float *) (out + t * out_stride))[j] = s >= 0 ? 1.0f : 0.0f;
            continue;
        }
        int v = s;
        if (s < 0) {
            for (;;) {
                if (spare >= n_gpu) break;
                bool used = false;
                for (int i = 0; i < k; ++i) used |= (slot_of(row[i]) == spare);
                if (!used) break;
                ++spare;
            }
            v = spare < n_gpu ? spare++ : 0;
        }
        ((int32_t *) (out + t * out_stride))[j] = v;
    }
}

__global__ void k_la_start(const float * x, const int32_t * ids, const float * w, int n_embd, int k, int n_tokens, int layer,
                           char * mb_dev, size_t off_x, size_t off_ids, size_t off_w, int * seq_dev, float * dst,
                           const int32_t * slot, int n_expert, int skip) {
    __shared__ int seq;
    __shared__ int n_cpu;
    if (threadIdx.x == 0) n_cpu = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < k * n_tokens; i += blockDim.x) {
        const int e = ids[i];
        if (e < 0 || e >= n_expert || slot[e] < 0) atomicAdd(&n_cpu, 1);
    }
    __syncthreads();
    if (n_cpu == 0 && skip) {
        // every expert of the job is in a slot: no job, and collect does not wait
        if (threadIdx.x == 0) ((int *) dst)[0] = -1;
        return;
    }
    if (threadIdx.x == 0) { seq = *seq_dev + 1; *seq_dev = seq; }
    __syncthreads();
    float   * mx = (float *)   (mb_dev + off_x);
    int32_t * mi = (int32_t *) (mb_dev + off_ids);
    float   * mw = (float *)   (mb_dev + off_w);
    for (int i = threadIdx.x; i < n_embd * n_tokens; i += blockDim.x) mx[i] = x[i];
    for (int i = threadIdx.x; i < k * n_tokens; i += blockDim.x) { mi[i] = ids[i]; mw[i] = w[i]; }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile ggml_moe_la_mailbox * mb = (volatile ggml_moe_la_mailbox *) mb_dev;
        mb->layer = layer; mb->n_tokens = n_tokens; mb->k = k; mb->n_embd = n_embd;
        __threadfence_system();
        mb->start_seq = seq;
        ((int *) dst)[0] = seq;
    }
}

__global__ void k_la_collect(const float * gpu_out, const float * start, char * mb_dev, size_t off_out, int n, float * dst) {
    __shared__ int ok;
    if (threadIdx.x == 0) {
        volatile ggml_moe_la_mailbox * mb = (volatile ggml_moe_la_mailbox *) mb_dev;
        const int seq = ((const int *) start)[0];
        if (seq < 0) { ok = 0; }     // no job: everything ran in slots
        const long long t0 = clock64();
        // about half a second at the clocks of current cards: a worker that is gone must not hang the GPU
        while (seq >= 0 && mb->done_seq != seq) {
            if (clock64() - t0 > 1000000000LL) { mb->error = 2; break; }
        }
        ok = seq >= 0 && mb->done_seq == seq;
        __threadfence_system();
    }
    __syncthreads();
    const volatile float * mo = (const volatile float *) (mb_dev + off_out);
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = gpu_out[i] + (ok ? mo[i] : 0.0f);
}

} // namespace

bool ggml_cuda_la_fast_ready(int device) {
    return g_fast.mb_host != nullptr && g_fast.device == device;
}

void ggml_cuda_la_fast_prepare(int device, int n_layers, int n_expert, int n_embd, int k, int max_tokens) {
    if (g_fast.mb_host) return;
    if (n_expert > LA_MAX_EXPERT || k > LA_MAX_K) {
        fprintf(stderr, "moe-la: fast path off (%d experts, %d per token: the kernels take at most %d and %d)\n",
                n_expert, k, LA_MAX_EXPERT, LA_MAX_K);
        return;
    }
    ggml_cuda_set_device(device);
    size_t off_x, off_ids, off_w, off_out;
    const size_t size = ggml_moe_la_mailbox_layout(n_embd, k, max_tokens, &off_x, &off_ids, &off_w, &off_out);
    // Our own memory, registered with CUDA rather than allocated by it: it is never freed, so the worker,
    // which watches it until the process ends, cannot read memory the CUDA runtime gave back at exit.
    const size_t page = 4096, bytes = (size + page - 1) / page * page;
#ifdef _WIN32
    void * host = _aligned_malloc(bytes, page);
#else
    void * host = nullptr;
    if (posix_memalign(&host, page, bytes) != 0) host = nullptr;
#endif
    if (!host) {
        fprintf(stderr, "moe-la: fast path off (no host memory for the mailbox)\n");
        return;
    }
    memset(host, 0, bytes);
    if (cudaHostRegister(host, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable) != cudaSuccess) {
        (void) cudaGetLastError();
        fprintf(stderr, "moe-la: fast path off (cannot map host memory)\n");
        return;
    }
    void * dev = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&dev, host, 0));
    CUDA_CHECK(cudaMalloc(&g_fast.seq_dev, sizeof(int)));
    CUDA_CHECK(cudaMemset(g_fast.seq_dev, 0, sizeof(int)));
    g_fast.slot.assign(n_layers, nullptr);
    g_fast.excess.assign(n_layers, nullptr);
    g_fast.n_gpu.assign(n_layers, 0);
    std::vector<int32_t> slot(n_expert);
    std::vector<float>   excess(n_expert);
    for (int il = 0; il < n_layers; ++il) {
        g_fast.n_gpu[il] = ggml_moe_la_fast_tables(il, n_expert, slot.data(), excess.data());
        CUDA_CHECK(cudaMalloc(&g_fast.slot[il],   sizeof(int32_t) * n_expert));
        CUDA_CHECK(cudaMalloc(&g_fast.excess[il], sizeof(float)   * n_expert));
        CUDA_CHECK(cudaMemcpy(g_fast.slot[il],   slot.data(),   sizeof(int32_t) * n_expert, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g_fast.excess[il], excess.data(), sizeof(float)   * n_expert, cudaMemcpyHostToDevice));
    }
    ggml_moe_la_fast_params(&g_fast.beta, &g_fast.gamma, &g_fast.protect);
    if (const char * v = getenv("LA_FAST_SKIP")) g_fast.skip = atoi(v) != 0;
    g_fast.device = device;
    g_fast.n_embd = n_embd; g_fast.k = k; g_fast.max_tokens = max_tokens; g_fast.n_expert = n_expert;
    g_fast.off_x = off_x; g_fast.off_ids = off_ids; g_fast.off_w = off_w; g_fast.off_out = off_out;
    g_fast.mb_dev  = (char *) dev;
    g_fast.mb_host = (ggml_moe_la_mailbox *) host;
    ggml_moe_la_fast_attach(g_fast.mb_host, n_embd, k, max_tokens);
}

void ggml_cuda_op_moe_la_fast(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int kind  = -dst->op_params[2];
    const int layer =  dst->op_params[3];
    cudaStream_t stream = ctx.stream();
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(g_fast.mb_host && layer >= 0 && layer < (int) g_fast.slot.size());
    switch (kind) {
        case 20: {
            GGML_ASSERT(src0->type == GGML_TYPE_F32 && src0->ne[0] == g_fast.n_expert);
            const int k = (int) dst->ne[0];
            k_la_select<<<(unsigned) src0->ne[1], WARP_SIZE, 0, stream>>>(
                (const float *) src0->data, src0->nb[1] / sizeof(float), (int) src0->ne[0], k,
                (int32_t *) dst->data, dst->nb[1] / sizeof(int32_t), g_fast.excess[layer],
                g_fast.beta, g_fast.gamma, g_fast.protect);
        } break;
        case 21:
        case 23: {
            const int n_tokens = (int) src0->ne[1];
            const int64_t out_stride = kind == 23 ? dst->nb[2] : dst->nb[1];
            k_la_map<<<(n_tokens + 63) / 64, 64, 0, stream>>>(
                (const int32_t *) src0->data, src0->nb[1] / sizeof(int32_t), (int) src0->ne[0], n_tokens, g_fast.n_expert,
                g_fast.slot[layer], g_fast.n_gpu[layer], kind - 20, (char *) dst->data, out_stride);
        } break;
        case 24: {
            const ggml_tensor * ids = dst->src[1];
            const ggml_tensor * w   = dst->src[2];
            GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(ids) && ggml_is_contiguous(w));
            const int n_tokens = (int) ids->ne[1], k = (int) ids->ne[0], n_embd = (int) src0->ne[0];
            GGML_ASSERT(n_tokens <= g_fast.max_tokens && k == g_fast.k && n_embd == g_fast.n_embd);
            k_la_start<<<1, 256, 0, stream>>>((const float *) src0->data, (const int32_t *) ids->data, (const float *) w->data,
                n_embd, k, n_tokens, layer, g_fast.mb_dev, g_fast.off_x, g_fast.off_ids, g_fast.off_w,
                g_fast.seq_dev, (float *) dst->data, g_fast.slot[layer], g_fast.n_expert, g_fast.skip);
        } break;
        case 25: {
            const ggml_tensor * start = dst->src[1];
            GGML_ASSERT(ggml_is_contiguous(src0));
            k_la_collect<<<1, 1024, 0, stream>>>((const float *) src0->data, (const float *) start->data, g_fast.mb_dev,
                g_fast.off_out, (int) ggml_nelements(dst), (float *) dst->data);
        } break;
        default:
            GGML_ABORT("moe-la fast: unknown kind %d", kind);
    }
    CUDA_CHECK(cudaGetLastError());
}
