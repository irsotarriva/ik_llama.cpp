#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct llama_model;
struct ggml_context;
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;

// Load-aware recipes: measured settings per model family, picked from the model's metadata.
// LA_RECIPES=dir (or one file) turns them on; LA_RECIPE_LEVEL picks a level (else the file's default).
// A recipe only sets LA_* variables that are not already set, so anything given by hand wins.
// Called once, after the metadata is read and before any tensor is loaded (the selector reads its
// settings when the first expert tensor is registered). See recipes/README.md.
void llama_la_apply_recipe(const llama_model & model, const std::string & fname);

// What the loader knows about each GPU of the model (one entry per model.devices), for the VRAM tier.
struct llama_la_gpus {
    std::vector<int>                        device;     // backend device id
    std::vector<ggml_backend_buffer_type_t> buft;       // its weight buffer type
    std::vector<std::string>                name;       // CUDA0, CUDA1, ...
    std::vector<std::string>                desc;       // the card's name
    std::vector<uint64_t>                   free;       // free memory before any weight is loaded
    std::vector<uint64_t>                   reserve;    // estimated KV cache and compute buffers on it
    std::vector<bool>                       display;    // drives a display (the driver may spill into RAM)
    int  main       = 0;                                // index of the main GPU
    bool offloading = false;                            // some layers run on a GPU (-ngl > 0)
};

// Whether and how the expert cache runs, decided once every weight has its buffer type (ctx_map) and
// before any is read or mapped.
// Without LA_ENABLE given by hand: LA_EXPERT_CACHE (auto, on, off; --expert-cache) and the budgets
// LA_EXPERTS_RAM (auto or a size; --experts-ram), LA_EXPERTS_VRAM (auto or a size per GPU, comma
// separated; --experts-vram) and LA_VRAM_MARGIN (--experts-vram-margin). In auto the cache stays off
// while the routed experts in host memory fit the RAM budget, and otherwise reads the rest from disk;
// it sets LA_ENABLE, LA_REAL, LA_RESIDENT_FRAC, LA_GPU_FRAC and LA_HEADROOM_MB.
// apply = false (--dry-run) works out the same choices without setting anything.
// The choices, with the reasons for the automatic ones, are kept for llama_expert_cache_config().
// Throws when a budget cannot be used (a VRAM budget on a GPU other than the main one).
void llama_la_decide(const llama_model & model, bool use_mmap,
                     const std::map<ggml_backend_buffer_type_t, ggml_context *> & ctx_map,
                     const llama_la_gpus & gpus, bool apply);

// The VRAM budgets given as sizes (--experts-vram), per GPU (0 where auto or none), when the cache is
// not off: --fit keeps that much free on each GPU.
std::vector<uint64_t> llama_la_vram_given(int n_gpus, int main_gpu);

// Whether the cache will read experts from disk (the same choice as llama_la_decide, without setting
// anything): the weights then have to stay memory-mapped.
bool llama_la_reads_disk(const llama_model & model, const std::map<ggml_backend_buffer_type_t, ggml_context *> & ctx_map);

// "24G", "24GB", "24GiB", "512M", "1.5T"; a plain number is GiB
bool llama_la_parse_size(const char * s, uint64_t & bytes);
