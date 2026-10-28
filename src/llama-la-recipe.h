#pragma once

#include <cstdint>
#include <map>
#include <string>

struct llama_model;
struct ggml_context;
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;

// Load-aware recipes: measured settings per model family, picked from the model's metadata.
// LA_RECIPES=dir (or one file) turns them on; LA_RECIPE_LEVEL picks a level (else the file's default).
// A recipe only sets LA_* variables that are not already set, so anything given by hand wins.
// Called once, after the metadata is read and before any tensor is loaded (the selector reads its
// settings when the first expert tensor is registered). See recipes/README.md.
void llama_la_apply_recipe(const llama_model & model, const std::string & fname);

// Whether and how the expert cache runs, decided once every weight has its buffer type (ctx_map) and
// before any is read or mapped.
// Without LA_ENABLE given by hand: LA_EXPERT_CACHE (auto, on, off; --expert-cache) and the budgets
// LA_EXPERTS_RAM (auto or a size; --experts-ram) and LA_EXPERTS_VRAM (--experts-vram). In auto the
// cache stays off while the routed experts in host memory fit the RAM budget, and otherwise reads the
// rest from disk; it sets LA_ENABLE, LA_REAL, LA_RESIDENT_FRAC, LA_GPU_FRAC and LA_HEADROOM_MB.
void llama_la_decide(const llama_model & model, bool use_mmap,
                     const std::map<ggml_backend_buffer_type_t, ggml_context *> & ctx_map);

// Whether the cache will read experts from disk (the same choice as llama_la_decide, without setting
// anything): the weights then have to stay memory-mapped.
bool llama_la_reads_disk(const llama_model & model, const std::map<ggml_backend_buffer_type_t, ggml_context *> & ctx_map);

// "24G", "24GB", "24GiB", "512M", "1.5T"; a plain number is GiB
bool llama_la_parse_size(const char * s, uint64_t & bytes);
