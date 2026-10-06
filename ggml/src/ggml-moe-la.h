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
//   LA_THETA=t             drop a selected expert whose probability is below
//                          t * top probability * price of its tier (default 0)
//   LA_PRICE_RESIDENT=p    price of a resident expert relative to a read one (default 0.1)
//   LA_PROFILE=path        float32 [n_layers * n_expert] routing frequencies; the
//                          most frequent experts start resident and popular layers
//                          get more slots. Without it every layer gets the same
//                          number of slots and starts empty.
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

#ifdef __cplusplus
}
#endif