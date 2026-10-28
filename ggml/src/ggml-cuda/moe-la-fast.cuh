#pragma once

#include "common.cuh"

// VRAM tier, fast path of the load-aware selector (see ggml-moe-la.h): the ARGSORT_THRESH nodes with
// op_params[2] = -20 .. -25.
void ggml_cuda_op_moe_la_fast(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// True once ggml_cuda_la_fast_prepare has made the mailbox and the tables for `device`.
bool ggml_cuda_la_fast_ready(int device);
