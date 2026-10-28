#include "llama-la-recipe.h"
#include "llama-model.h"
#include "llama-arch.h"
#include "llama-impl.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace {

// The LA_* settings that decide whether and how the expert cache runs, as found before any of this
// code set one: given by hand, they switch the automatic choice off (see llama_la_decide).
struct by_hand {
    bool noted = false, enable = false, real = false, frac = false, gpu = false, headroom = false;
};

by_hand & hand() {
    static by_hand h;
    if (!h.noted) {
        h.noted    = true;
        h.enable   = getenv("LA_ENABLE")        != nullptr;
        h.real     = getenv("LA_REAL")          != nullptr;
        h.frac     = getenv("LA_RESIDENT_FRAC") != nullptr;
        h.gpu      = getenv("LA_GPU_FRAC")      != nullptr;
        h.headroom = getenv("LA_HEADROOM_MB")   != nullptr;
    }
    return h;
}

struct recipe {
    fs::path path;
    std::map<std::string, std::string> match;           // arch, n_expert, n_expert_used, name
    std::string default_level;
    std::vector<std::pair<std::string, std::string>> common;   // set at every level
    std::map<std::string, std::vector<std::pair<std::string, std::string>>> levels;
    std::vector<std::string> level_order;
};

std::string trim(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

std::string lower(std::string s) {
    for (char & c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

// Format (one recipe per file, *.recipe):
//   # comment
//   match arch=qwen35moe n_expert=256 name=Qwen3.6     (all given keys must hold; name: substring
//                                                       of general.name or the file name, any case)
//   default balanced
//   LA_ENABLE=1                                        (before the first [level]: every level)
//   [exact]
//   LA_BETA=0
//   [balanced]
//   LA_BETA=0.25   # trailing comments are allowed
bool parse(const fs::path & p, recipe & r) {
    std::ifstream f(p);
    if (!f) return false;
    r.path = p;
    std::string line, cur;
    while (std::getline(f, line)) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;
        if (line.front() == '[' && line.back() == ']') {
            cur = trim(line.substr(1, line.size() - 2));
            if (!r.levels.count(cur)) r.level_order.push_back(cur);
            r.levels[cur];
            continue;
        }
        if (line.rfind("match", 0) == 0 && cur.empty()) {
            std::istringstream ss(line.substr(5));
            std::string kv;
            while (ss >> kv) {
                const size_t eq = kv.find('=');
                if (kv == "*") continue;
                if (eq == std::string::npos) return false;
                r.match[kv.substr(0, eq)] = kv.substr(eq + 1);
            }
            continue;
        }
        if (line.rfind("default", 0) == 0 && cur.empty() && line.find('=') == std::string::npos) {
            r.default_level = trim(line.substr(7));
            continue;
        }
        const size_t eq = line.find('=');
        if (eq == std::string::npos) return false;
        auto kv = std::make_pair(trim(line.substr(0, eq)), trim(line.substr(eq + 1)));
        if (cur.empty()) r.common.push_back(kv); else r.levels[cur].push_back(kv);
    }
    if (r.default_level.empty() && !r.level_order.empty()) r.default_level = r.level_order.front();
    return true;
}

// -1: does not match; otherwise how many keys matched (more is more specific)
int score(const recipe & r, const llama_model & model, const std::string & fname) {
    int n = 0;
    for (const auto & [k, v] : r.match) {
        bool ok = false;
        if (k == "arch")               ok = lower(v) == lower(llama_model_arch_name(model.arch));
        else if (k == "n_expert")      ok = std::to_string(model.hparams.n_expert) == v;
        else if (k == "n_expert_used") ok = std::to_string(model.hparams.n_expert_used) == v;
        else if (k == "name")          ok = lower(model.name).find(lower(v)) != std::string::npos ||
                                            lower(fs::path(fname).filename().string()).find(lower(v)) != std::string::npos;
        if (!ok) return -1;
        ++n;
    }
    return n;
}

void set_env(const std::string & k, const std::string & v) {
#ifdef _WIN32
    _putenv_s(k.c_str(), v.c_str());
#else
    setenv(k.c_str(), v.c_str(), 1);
#endif
}

} // namespace

bool llama_la_parse_size(const char * s, uint64_t & bytes) {
    if (!s) return false;
    char * end = nullptr;
    const double v = strtod(s, &end);
    if (end == s || !(v >= 0) || !std::isfinite(v)) return false;
    std::string unit = lower(trim(end));
    if (!unit.empty() && unit.back() == 'b') unit.pop_back();       // "GB", "GiB" -> "g", "gi"
    if (unit.size() == 2 && unit[1] == 'i') unit.pop_back();
    double mul;
    if      (unit.empty() || unit == "g") mul = 1024.0 * 1024.0 * 1024.0;   // a plain number is GiB
    else if (unit == "m")                 mul = 1024.0 * 1024.0;
    else if (unit == "t")                 mul = 1024.0 * 1024.0 * 1024.0 * 1024.0;
    else return false;
    bytes = (uint64_t) (v * mul);
    return true;
}

void llama_la_apply_recipe(const llama_model & model, const std::string & fname) {
    static std::atomic<bool> done{false};
    hand();                                                  // before a recipe sets anything
    const char * mode = getenv("LA_EXPERT_CACHE");
    if (mode && lower(mode) == "off") return;
    const char * where = getenv("LA_RECIPES");
#ifdef LLAMA_LA_RECIPES_DIR
    // the recipes of the source tree this was built from, when nothing else is given
    static const std::string built_in = LLAMA_LA_RECIPES_DIR;
    std::error_code ec0;
    if ((!where || !*where) && fs::is_directory(fs::path(built_in), ec0)) where = built_in.c_str();
#endif
    if (!where || !*where || done.exchange(true)) return;   // the first model loaded owns the selector

    std::vector<recipe> all;
    std::error_code ec;
    const fs::path base(where);
    if (fs::is_directory(base, ec)) {
        for (const auto & e : fs::directory_iterator(base, ec)) {
            if (e.path().extension() != ".recipe") continue;
            recipe r;
            if (parse(e.path(), r)) all.push_back(std::move(r));
            else LLAMA_LOG_WARN("la-recipe: cannot read %s\n", e.path().string().c_str());
        }
    } else {
        recipe r;
        if (parse(base, r)) all.push_back(std::move(r));
        else LLAMA_LOG_WARN("la-recipe: cannot read %s\n", where);
    }

    const recipe * best = nullptr;
    int best_score = -1;
    for (const auto & r : all) {
        const int s = score(r, model, fname);
        // equal scores: the file name decides, so the choice does not depend on the directory order
        if (s > best_score || (s == best_score && best && r.path.filename() < best->path.filename())) {
            best = &r; best_score = s;
        }
    }
    if (!best) {
        LLAMA_LOG_INFO("la-recipe: no recipe in %s matches arch=%s n_expert=%u name=%s\n", where,
                       llama_model_arch_name(model.arch), model.hparams.n_expert, model.name.c_str());
        return;
    }

    // without a level asked for, exact: the selector never serves another expert than the router's
    std::string level = best->levels.count("exact") ? "exact" : best->default_level;
    if (const char * v = getenv("LA_RECIPE_LEVEL")) level = v;
    if (!level.empty() && !best->levels.count(level)) {
        std::string known;
        for (const auto & l : best->level_order) known += (known.empty() ? "" : ", ") + l;
        LLAMA_LOG_WARN("la-recipe: %s has no level '%s' (levels: %s); using '%s'\n",
                       best->path.filename().string().c_str(), level.c_str(), known.c_str(), best->default_level.c_str());
        level = best->default_level;
    }

    std::vector<std::pair<std::string, std::string>> vars = best->common;
    if (best->levels.count(level)) {
        const auto & l = best->levels.at(level);
        vars.insert(vars.end(), l.begin(), l.end());
    }
    std::string applied, kept;
    for (auto [k, v] : vars) {
        if (k == "LA_ENABLE") continue;   // whether the cache runs is decided at load (llama_la_decide)
        // paths in a recipe are relative to the recipe's folder
        if ((k == "LA_PROFILE" || k == "LA_ROW_PROFILE") && !v.empty() && fs::path(v).is_relative()) {
            v = (best->path.parent_path() / v).string();
        }
        if (getenv(k.c_str())) { kept += " " + k; continue; }
        set_env(k, v);
        applied += " " + k + "=" + v;
    }
    LLAMA_LOG_INFO("la-recipe: %s, level '%s':%s\n", best->path.filename().string().c_str(), level.c_str(),
                   applied.empty() ? " (nothing new)" : applied.c_str());
    if (!kept.empty()) LLAMA_LOG_INFO("la-recipe: given by hand, kept:%s\n", kept.c_str());
}

namespace {

// bytes of the routed experts and of the other weights that will live in host memory
void count_host(const llama_model & model, const std::map<ggml_backend_buffer_type_t, ggml_context *> & ctx_map,
                uint64_t & experts, uint64_t & others, bool & merged) {
    std::vector<const ggml_tensor *> routed, merged_up_gate;
    for (const auto & l : model.layers) {
        for (const ggml_tensor * t : { l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps, l.ffn_up_gate_exps }) {
            if (t) routed.push_back(t);
        }
        if (l.ffn_up_gate_exps) merged_up_gate.push_back(l.ffn_up_gate_exps);
    }
    experts = others = 0;
    merged = false;
    for (const auto & bc : ctx_map) {
        if (!bc.first || !bc.second || !ggml_backend_buft_is_host(bc.first)) continue;
        for (ggml_tensor * t = ggml_get_first_tensor(bc.second); t; t = ggml_get_next_tensor(bc.second, t)) {
            if (std::find(routed.begin(), routed.end(), t) != routed.end()) {
                experts += ggml_nbytes(t);
                merged = merged || std::find(merged_up_gate.begin(), merged_up_gate.end(), t) != merged_up_gate.end();
            } else {
                others += ggml_nbytes(t);
            }
        }
    }
}

// RAM for the routed experts: --experts-ram, or in auto what the system can give now, less the other
// weights and a reserve for the system, the KV cache and other programs (the larger of 10% and 2 GiB)
bool ram_budget(uint64_t others, uint64_t & ram, uint64_t & reserve, bool & is_auto) {
    const char * s = getenv("LA_EXPERTS_RAM");
    is_auto = !s || lower(s) == "auto";
    reserve = 0;
    if (!is_auto) return llama_la_parse_size(s, ram);
    const uint64_t avail = ggml_moe_la_mem_available(), total = ggml_moe_la_mem_total();
    if (avail == 0) return false;
    reserve = std::max<uint64_t>(2ull << 30, total / 10);
    ram = avail > others + reserve ? avail - others - reserve : 0;
    return true;
}

std::string cache_mode() {
    const char * m = getenv("LA_EXPERT_CACHE");
    return m ? lower(m) : "auto";
}

} // namespace

bool llama_la_reads_disk(const llama_model & model, const std::map<ggml_backend_buffer_type_t, ggml_context *> & ctx_map) {
    const by_hand & h = hand();
    if (h.enable) {
        const char * e = getenv("LA_ENABLE"), * r = getenv("LA_REAL");
        return e && atoi(e) != 0 && r && atoi(r) != 0;
    }
    if (cache_mode() == "off" || model.hparams.n_expert == 0) return false;
    uint64_t experts, others, ram, reserve;
    bool merged, is_auto;
    count_host(model, ctx_map, experts, others, merged);
    return experts > 0 && !merged && ram_budget(others, ram, reserve, is_auto) && experts > ram;
}

void llama_la_decide(const llama_model & model, bool use_mmap,
                     const std::map<ggml_backend_buffer_type_t, ggml_context *> & ctx_map) {
    static std::atomic<bool> done{false};
    if (done.exchange(true)) return;                         // the first model loaded owns the selector
    const by_hand & h = hand();
    if (h.enable) {
        LLAMA_LOG_INFO("expert cache: LA_ENABLE given, using the LA_* settings as they are\n");
        return;
    }
    std::string mode = cache_mode();
    if (mode != "auto" && mode != "on" && mode != "off") {
        LLAMA_LOG_WARN("expert cache: unknown mode '%s', using auto\n", mode.c_str());
        mode = "auto";
    }
    auto off = [](const char * why) {
        set_env("LA_ENABLE", "0");
        if (why) LLAMA_LOG_INFO("expert cache: off (%s)\n", why);
    };
    if (mode == "off") { off("--expert-cache off"); return; }
    if (model.hparams.n_expert == 0 || model.hparams.n_expert_used == 0) { off(nullptr); return; }

    uint64_t experts, others;
    bool merged;
    count_host(model, ctx_map, experts, others, merged);
    if (experts == 0) { off("no routed experts in host memory"); return; }

    const double GiB = 1024.0 * 1024.0 * 1024.0;
    uint64_t ram = 0, vram = 0, reserve = 0;
    bool ram_auto;
    if (!ram_budget(others, ram, reserve, ram_auto)) {
        off(ram_auto ? "this system does not say how much memory is free; give --experts-ram" : "cannot read --experts-ram");
        return;
    }
    if (const char * v = getenv("LA_EXPERTS_VRAM")) {
        if (!llama_la_parse_size(v, vram)) LLAMA_LOG_WARN("expert cache: cannot read --experts-vram '%s', no VRAM tier\n", v);
    }

    const bool needs_disk = experts > ram;
    if (!needs_disk && vram == 0 && mode != "on") {
        LLAMA_LOG_INFO("expert cache: off, not needed (%.1f GiB of experts, %.1f GiB of RAM for them%s)\n",
                       experts / GiB, ram / GiB, ram_auto ? ", auto" : "");
        off(nullptr);
        return;
    }
    if (needs_disk && (!use_mmap || merged)) {
        off(!use_mmap ? "the experts do not fit in RAM, but reading them from disk needs mmap (drop --no-mmap)"
                      : "the experts do not fit in RAM, but merged up/gate expert tensors cannot be read from disk");
        return;
    }

    char buf[64];
    set_env("LA_ENABLE", "1");
    if (!h.real) set_env("LA_REAL", needs_disk ? "1" : "0");
    // a layer needs room for one token's experts
    const double min_frac = (double) model.hparams.n_expert_used / model.hparams.n_expert;
    double frac = needs_disk ? (double) ram / experts : 1.0;
    if (frac < min_frac) {
        LLAMA_LOG_WARN("expert cache: %.1f GiB of RAM is less than one token's experts in every layer; using %.1f GiB\n",
                       ram / GiB, min_frac * experts / GiB);
        frac = min_frac;
    }
    if (!h.frac) { snprintf(buf, sizeof(buf), "%.6f", frac); set_env("LA_RESIDENT_FRAC", buf); }
    const double gpu_frac = std::min(1.0, (double) vram / experts);
    if (!h.gpu && vram > 0) { snprintf(buf, sizeof(buf), "%.6f", gpu_frac); set_env("LA_GPU_FRAC", buf); }
    if (needs_disk && ram_auto && !h.headroom) {
        snprintf(buf, sizeof(buf), "%" PRIu64, reserve / (1024 * 1024));
        set_env("LA_HEADROOM_MB", buf);
    }
    const char * level = getenv("LA_RECIPE_LEVEL");
    LLAMA_LOG_INFO("expert cache: on, %.1f GiB of experts: %.1f GiB in RAM%s, %.1f GiB in VRAM, %s; quality %s\n",
                   experts / GiB, frac * experts / GiB, ram_auto ? " (auto)" : "",
                   gpu_frac * experts / GiB, needs_disk ? "the rest read from disk" : "all in memory",
                   level ? level : "exact");
}
