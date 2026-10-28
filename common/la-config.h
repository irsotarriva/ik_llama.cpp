#pragma once

// The engine's config file (--config file.toml): the expert cache and the server's disk prompt cache.
//
// Every key is optional; a key that is missing is chosen at load (auto). The settings are the ones of
// the command-line flags, grouped in sections:
//
//   [experts]
//   cache   = "auto"         # --expert-cache   auto | on | off
//   quality = "exact"        # --quality        exact | near | balanced | fast
//   ram     = "auto"         # --experts-ram    auto | size ("24G", "512M"; a number is GiB)
//
//   [devices.CUDA0]          # one section per GPU, named as the backend names it
//   vram        = "auto"     # --experts-vram        auto | size
//   vram_margin = "1G"       # --experts-vram-margin size
//
//   [cache_disk]
//   dir = "prompt-cache"     # --cache-disk      (relative to the file's folder)
//   max = "20G"              # --cache-disk-max
//
// Precedence: automatic choices < this file < command-line flags. The LA_* variables the flags set
// are the engine's interface: one already set in the environment is kept as well, and LA_ENABLE set
// by hand turns the automatic choice off altogether.
//
// The file is read into common_la_config, which also has a JSON form (the same layout) so that a
// server can take and return the config over HTTP.

#include <nlohmann/json_fwd.hpp>

#include <map>
#include <optional>
#include <string>

struct common_la_config {
    struct device {
        std::optional<std::string> vram;          // "auto" or a size
        std::optional<std::string> vram_margin;   // a size
    };

    // [experts]
    std::optional<std::string> cache;             // auto | on | off
    std::optional<std::string> quality;           // exact | near | balanced | fast
    std::optional<std::string> ram;               // "auto" or a size

    // [devices.<name>], by name ("CUDA0")
    std::map<std::string, device> devices;

    // [cache_disk]
    std::optional<std::string> cache_disk_dir;
    std::optional<std::string> cache_disk_max;    // a size
};

// Reads a config file: TOML, or JSON when the name ends in .json. Relative paths in it are made
// relative to the file's folder. Returns false with a message naming the file and key on error.
bool common_la_config_load(const std::string & path, common_la_config & cfg, std::string & err);

// The JSON form. Unknown keys are errors; "notes" and "summary" (written by --print-config) are skipped.
bool common_la_config_from_json(const nlohmann::ordered_json & j, common_la_config & cfg, std::string & err);
nlohmann::ordered_json common_la_config_to_json(const common_la_config & cfg);

// Sets the LA_* variables for the keys the config gives, except those already set (by a flag, or
// by hand in the environment).
void common_la_config_apply(const common_la_config & cfg);

// The resolved config (llama_expert_cache_config(), plus the disk prompt cache's settings) as a TOML
// file with every automatic choice filled in and a comment on how it was chosen.
std::string common_la_config_resolved_toml(const std::string & model_path);

// The resolved config as JSON, in the same layout.
nlohmann::ordered_json common_la_config_resolved_json();

// "24G", "24GB", "24GiB", "512M", "1.5T"; a plain number is GiB
bool common_la_size_ok(const std::string & s);
