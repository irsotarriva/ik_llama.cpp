#pragma once

// Load-aware expert selection for MoE layers (experimental).
//
// Replaces the plain top-k over router probabilities with a selection that
// knows which experts are resident, and optionally owns the memory the routed
// experts live in.
//
// Two modes:
//   modelled  residency is a model kept by the selector (a fixed budget of
//             experts, least-recently-served eviction). Weights stay in the
//             memory-mapped file. Measures quality and counts the reads a
//             rule would cause.
//   real      (LA_REAL=1) each routed-expert tensor is moved to a reserved
//             address range of its own. Only resident experts are backed by
//             memory; serving another one reads it from the file with
//             unbuffered I/O and evicting one releases its pages. The file is
//             only ever read, nothing is paged out, and memory use is the budget.
//
// Configuration is read once from the environment:
//   LA_ENABLE=1            turn the selector on
//   LA_REAL=1              real mode (needs mmap, CPU expert tensors, no up/gate merge)
//   LA_RESIDENT_FRAC=f     share of all routed experts that fit in memory (default 1)
//   LA_BETA=b              bias against non-resident experts, in units of the
//                          probability range of the router top-k (default 0)
//   LA_GAMMA=g             multiplicative bias: a non-resident expert competes with its
//                          probability times exp(-g), so only experts of similar probability
//                          replace each other (default 0)
//   LA_PREFETCH=m          while a layer computes, apply the next layer's router to this layer's
//                          input and start reading its m best guesses that are not in memory
//   LA_PROTECT=m           the router's first m experts are always served (default 0)
//   LA_THETA=t             drop a selected expert whose probability is below
//                          t * top probability * price of its tier (default 0)
//   LA_PRICE_RESIDENT=p    price of a resident expert relative to a read one (default 0.1)
//   LA_PROFILE=path        float32 [n_layers * n_expert] routing frequencies; the
//                          most frequent experts start resident and popular layers
//                          get more slots. Without it every layer gets the same
//                          number of slots and starts empty.
//   LA_GPU_FRAC=g          share of all routed experts kept in VRAM slots (needs LA_PROFILE
//                          and a GPU backend); they are the most frequent ones and never
//                          occupy host memory
//   LA_EXCESS_RAM=x        bias against an expert in host memory relative to one in VRAM,
//                          on the scale where a non-resident expert is 1 (default 0)
//   LA_ASYNC=1             an expert the router asked for but the selection replaced is read in
//                          the background and becomes resident the next time its layer runs,
//                          so the cache follows what the router wants without making tokens wait
//   LA_ASYNC_QUEUE=n       real mode: most experts waiting for a background read (default 64)
//   LA_ASYNC_THREADS=n     real mode: background reader threads (default 2)
//   LA_ASYNC_RATE=x        modelled mode: background reads per layer per token (default 1.5)
//   LA_EVICT=freq|lru      which resident expert makes room: the one the router asked for least
//                          (decayed count, the default) or the least recently served
//   LA_DECAY=d             factor applied to the request counts every 16 tokens (default 0.98)
//   LA_IO_THREADS=n        reader threads in real mode (default 8)
//   LA_STATS=path          append one JSON line of counters at exit

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;

bool ggml_moe_la_enabled(void);
bool ggml_moe_la_real(void);
// True if `data` is the memory of a tensor taken over in real mode. Such a tensor is only
// partly backed and must never be read as a whole.
bool ggml_moe_la_owns(const void * data);

// Real mode, at model load: take over one routed-expert tensor of `layer`
// ([n_in, n_out, n_expert], stored at `file_offset` of `path`). Changes
// tensor->data. Returns false (and leaves the tensor alone) if it cannot.
bool ggml_moe_la_register(int layer, struct ggml_tensor * tensor, const char * path, uint64_t file_offset);
// Real mode, after every tensor is registered: read the starting resident set.
void ggml_moe_la_prefill(void);

// A run of consecutive tokens of one layer: begin, one select per token, end.
// In real mode `end` returns once every selected expert is in memory.
void ggml_moe_la_begin(int layer);
// probs: n_expert router probabilities. ids: n_expert entries out; the
// selection comes first (most probable first), every other entry is -1.
// At most k experts are selected, at least one.
void ggml_moe_la_select(int layer, const float * probs, int n_expert, int k, int32_t * ids);
void ggml_moe_la_end(int layer);

bool ggml_moe_la_prefetching(void);
// logits: the router of `layer` applied to the input of the layer before it (n_expert values)
void ggml_moe_la_prefetch(int layer, const float * logits, int n_expert);

// VRAM tier. The engine keeps a copy of some experts of each layer in GPU slot
// tensors and runs them there; the rest run on the CPU as usual.
bool ggml_moe_la_gpu(void);
// Which experts of `layer` get a slot: writes up to max_slots expert ids, returns the count.
int  ggml_moe_la_gpu_plan(int layer, int n_expert, int32_t * experts, int max_slots);
// The engine could not create the slots of `layer`: its experts go back to the CPU.
void ggml_moe_la_gpu_cancel(int layer);
// Split one token's selection (k entries, -1 = dropped) between the two sides.
//   kind 1: int32 slot index of each entry that runs on the GPU; the others get the index one past
//           the last slot, where the engine keeps an all-zero expert so they contribute nothing
//   kind 2: int32 expert id of each entry that runs on the CPU, -1 for the others
//   kind 3: float 1 for each entry that runs on the GPU, 0 for the others
void ggml_moe_la_aux(int layer, int kind, const int32_t * ids, int k, void * out);

#ifdef __cplusplus
}
#endif