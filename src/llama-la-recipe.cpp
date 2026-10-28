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
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
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

// The share of (layer, expert) pairs to put in GPU slots for a budget of `bytes`. The selector ranks
// the pairs by the usage profile and gives slots to the first gpu_frac of them (ggml-moe-la.cpp,
// plan()); experts differ in size between layers (a Q4_K_M model has some down projections in Q6_K),
// so the share is found by walking that ranking and adding up sizes, not by dividing by the average.
// Layers whose experts are not in host memory get no slots but still take their place in the ranking.
double gpu_frac_for(const llama_model & model, const std::map<ggml_backend_buffer_type_t, ggml_context *> & ctx_map,
                    uint64_t bytes, uint64_t experts) {
    const double fallback = experts > 0 ? std::min(1.0, (double) bytes / experts) : 0.0;
    const char * path = getenv("LA_PROFILE");
    const int n_expert = (int) model.hparams.n_expert;
    if (!path || !*path || n_expert <= 0) return fallback;
    std::vector<float> profile;
    if (FILE * f = fopen(path, "rb")) {
        fseek(f, 0, SEEK_END);
        const long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        profile.resize(n > 0 ? (size_t) n / sizeof(float) : 0);
        if (fread(profile.data(), sizeof(float), profile.size(), f) != profile.size()) profile.clear();
        fclose(f);
    }
    if (profile.empty() || profile.size() % n_expert != 0) return fallback;
    std::vector<const ggml_tensor *> host;
    for (const auto & bc : ctx_map) {
        if (!bc.first || !bc.second || !ggml_backend_buft_is_host(bc.first)) continue;
        for (ggml_tensor * t = ggml_get_first_tensor(bc.second); t; t = ggml_get_next_tensor(bc.second, t)) host.push_back(t);
    }
    const int n_layers = (int) (profile.size() / n_expert);
    std::vector<uint64_t> per_expert(n_layers, 0);       // bytes of one expert of each layer
    for (int il = 0; il < n_layers && il < (int) model.layers.size(); ++il) {
        const auto & l = model.layers[il];
        uint64_t b = 0;
        bool ok = true;
        for (const ggml_tensor * t : { l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps }) {
            ok = ok && t && std::find(host.begin(), host.end(), t) != host.end();
            if (t) b += ggml_nbytes(t) / (uint64_t) t->ne[2];
        }
        per_expert[il] = ok ? b : 0;
    }
    std::vector<int> order(profile.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = (int) i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return profile[a] > profile[b]; });
    uint64_t sum = 0;
    size_t n = 0;
    for (; n < order.size(); ++n) {
        const uint64_t b = per_expert[order[n] / n_expert];
        if (sum + b > bytes) break;
        sum += b;
    }
    // the selector takes floor(gpu_frac * size): aim between n and n + 1 so rounding gives n
    return n >= order.size() ? 1.0 : ((double) n + 0.5) / (double) order.size();
}

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
bool ram_budget(uint64_t others, uint64_t & ram, uint64_t & reserve, bool & is_auto, uint64_t * avail_out = nullptr) {
    const char * s = getenv("LA_EXPERTS_RAM");
    is_auto = !s || lower(s) == "auto";
    reserve = 0;
    if (!is_auto) return llama_la_parse_size(s, ram);
    const uint64_t avail = ggml_moe_la_mem_available(), total = ggml_moe_la_mem_total();
    if (avail_out) *avail_out = avail;
    if (avail == 0) return false;
    reserve = std::max<uint64_t>(2ull << 30, total / 10);
    ram = avail > others + reserve ? avail - others - reserve : 0;
    return true;
}

std::string cache_mode() {
    const char * m = getenv("LA_EXPERT_CACHE");
    return m ? lower(m) : "auto";
}

std::vector<std::string> split_list(const char * s) {
    std::vector<std::string> out;
    if (!s) return out;
    std::string cur;
    for (const char * p = s; ; ++p) {
        if (*p == ',' || *p == 0) {
            out.push_back(lower(trim(cur)));
            cur.clear();
            if (!*p) break;
        } else {
            cur += *p;
        }
    }
    return out;
}

// One entry per GPU: -1 = auto, otherwise bytes. A single value is the main GPU's (the others get 0,
// as before budgets per GPU); a list goes by device order, and a GPU it does not reach is auto. An
// item can also name its GPU, "cuda1=2g" (the config file's form): the number at the end of the name
// is the device's place.
bool per_gpu(const char * env, int n_gpus, int main_gpu, std::vector<int64_t> & out, std::string & err) {
    out.assign(n_gpus, -1);
    const auto items = split_list(getenv(env));
    if (items.empty() || n_gpus == 0) return true;
    bool named = false;
    for (const auto & v : items) named = named || v.find('=') != std::string::npos;
    if ((int) items.size() > n_gpus && !named) {
        err = std::to_string(items.size()) + " values for " + std::to_string(n_gpus) + " GPU" + (n_gpus == 1 ? "" : "s");
        return false;
    }
    auto value = [&](const std::string & v, int64_t & b) {
        if (v.empty() || v == "auto") { b = -1; return true; }
        uint64_t u;
        if (!llama_la_parse_size(v.c_str(), u)) { err = "cannot read '" + v + "'"; return false; }
        b = (int64_t) u;
        return true;
    };
    if (named) {
        for (const auto & item : items) {
            const size_t eq = item.find('=');
            if (eq == std::string::npos) { err = "'" + item + "' among named values (give name=size for each)"; return false; }
            const std::string name = trim(item.substr(0, eq));
            size_t d = name.size();
            while (d > 0 && std::isdigit((unsigned char) name[d - 1])) --d;
            if (d == name.size()) { err = "no device number in '" + name + "'"; return false; }
            const int idx = atoi(name.c_str() + d);
            if (idx >= n_gpus) {
                err = "no device " + name + ": the model has " + std::to_string(n_gpus) + " GPU" + (n_gpus == 1 ? "" : "s") +
                      " (numbered from 0)";
                return false;
            }
            if (!value(trim(item.substr(eq + 1)), out[idx])) return false;
        }
        return true;
    }
    if (items.size() == 1 && n_gpus > 1) {
        for (int i = 0; i < n_gpus; ++i) out[i] = 0;
        return value(items[0], out[main_gpu]);
    }
    for (int i = 0; i < (int) items.size(); ++i) {
        if (!value(items[i], out[i])) return false;
    }
    return true;
}

// a size as the flags take it: "12.5G", "512M", "0"
std::string fmt_size(uint64_t b) {
    if (b == 0) return "0";
    char buf[32];
    if (b < (1ull << 30)) {
        snprintf(buf, sizeof(buf), "%.0fM", b / (1024.0 * 1024.0));
        return buf;
    }
    snprintf(buf, sizeof(buf), "%.2f", b / (1024.0 * 1024.0 * 1024.0));
    std::string s = buf;                                      // 12.50 -> 12.5, 12.00 -> 12
    while (s.back() == '0') s.pop_back();
    if (s.back() == '.') s.pop_back();
    return s + "G";
}

std::string json_str(const std::string & v) {
    std::string o = "\"";
    for (char c : v) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if ((unsigned char) c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
        else o += c;
    }
    return o + "\"";
}

using kv_list = std::vector<std::pair<std::string, std::string>>;

std::string json_obj(const kv_list & kv) {
    std::string o = "{";
    for (size_t i = 0; i < kv.size(); ++i) o += (i ? "," : "") + json_str(kv[i].first) + ":" + json_str(kv[i].second);
    return o + "}";
}

// The resolved settings in the config file's layout, with a note on how each automatic one was chosen
// ("experts.ram" -> "auto: ..."). Kept as JSON for llama_expert_cache_config().
struct resolved {
    kv_list experts;
    struct gpu { std::string name, desc; kv_list kv; };
    std::vector<gpu> gpus;
    kv_list notes;
    std::string summary;

    std::string json() const {
        std::string o = "{\"experts\":" + json_obj(experts) + ",\"devices\":{";
        for (size_t g = 0; g < gpus.size(); ++g) o += (g ? "," : "") + json_str(gpus[g].name) + ":" + json_obj(gpus[g].kv);
        kv_list all = notes;
        for (const auto & g : gpus) if (!g.desc.empty()) all.push_back({ "devices." + g.name, g.desc });
        return o + "},\"notes\":" + json_obj(all) + ",\"summary\":" + json_str(summary) + "}";
    }
};

std::string & resolved_json() {
    static std::string s;
    return s;
}

} // namespace

const char * llama_expert_cache_config(void) {
    return resolved_json().c_str();
}

std::vector<uint64_t> llama_la_vram_given(int n_gpus, int main_gpu) {
    std::vector<uint64_t> out(std::max(n_gpus, 0), 0);
    if (n_gpus <= 0 || hand().enable || cache_mode() == "off") return out;
    std::vector<int64_t> v;
    std::string err;
    if (!per_gpu("LA_EXPERTS_VRAM", n_gpus, std::clamp(main_gpu, 0, n_gpus - 1), v, err)) return out;
    for (int i = 0; i < n_gpus; ++i) out[i] = v[i] > 0 ? (uint64_t) v[i] : 0;
    return out;
}

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
                     const std::map<ggml_backend_buffer_type_t, ggml_context *> & ctx_map,
                     const llama_la_gpus & gpus, bool apply) {
    static std::atomic<bool> done{false};
    if (done.exchange(true)) return;                         // the first model loaded owns the selector
    const double GiB = 1024.0 * 1024.0 * 1024.0;
    const by_hand & h = hand();
    const int n_gpus   = (int) gpus.device.size();
    const int main_gpu = n_gpus > 0 ? std::clamp(gpus.main, 0, n_gpus - 1) : 0;
    const char * level_env = getenv("LA_RECIPE_LEVEL");
    const std::string level = level_env ? lower(level_env) : "exact";

    resolved R;
    for (int i = 0; i < n_gpus; ++i) R.gpus.push_back({ gpus.name[i], gpus.desc[i], {} });
    std::string ram_value;                                   // empty: not worked out
    auto setv = [apply](const char * k, const std::string & v) { if (apply) set_env(k, v); };
    auto finish = [&](const char * cache, const std::string & why, const std::string & summary) {
        R.experts = { { "cache", cache }, { "quality", level } };
        if (!ram_value.empty()) R.experts.push_back({ "ram", ram_value });
        if (!why.empty()) R.notes.insert(R.notes.begin(), { "experts.cache", why });
        R.summary = summary;
        resolved_json() = R.json();
    };

    if (h.enable) {
        LLAMA_LOG_INFO("expert cache: LA_ENABLE given, using the LA_* settings as they are\n");
        const char * e = getenv("LA_ENABLE");
        finish(e && atoi(e) ? "on" : "off", "LA_ENABLE given by hand: the LA_* variables are used as they are",
               "LA_ENABLE given by hand");
        return;
    }
    std::string mode = cache_mode();
    if (mode != "auto" && mode != "on" && mode != "off") {
        LLAMA_LOG_WARN("expert cache: unknown mode '%s', using auto\n", mode.c_str());
        mode = "auto";
    }
    const std::string by = mode == "auto" ? "auto: " : "";
    auto off = [&](const std::string & why, bool log = true) {
        setv("LA_ENABLE", "0");
        if (log) LLAMA_LOG_INFO("expert cache: off (%s)\n", why.c_str());
        for (auto & g : R.gpus) g.kv = { { "vram", "0" } };
        finish("off", by + why, "off (" + why + ")");
    };
    if (mode == "off") { off("--expert-cache off"); return; }
    if (model.hparams.n_expert == 0 || model.hparams.n_expert_used == 0) { off("the model has no routed experts", false); return; }

    uint64_t experts, others;
    bool merged;
    count_host(model, ctx_map, experts, others, merged);
    if (experts == 0) { off("no routed experts in host memory"); return; }

    uint64_t ram = 0, reserve = 0, avail = 0;
    bool ram_auto;
    if (!ram_budget(others, ram, reserve, ram_auto, &avail)) {
        off(ram_auto ? "this system does not say how much memory is free; give --experts-ram" : "cannot read --experts-ram");
        return;
    }
    ram_value = fmt_size(ram);
    if (ram_auto) {
        char b[256];
        snprintf(b, sizeof(b), "auto: %.1f GiB free, less %.1f GiB of other weights and a reserve of %.1f GiB",
                 avail / GiB, others / GiB, reserve / GiB);
        R.notes.push_back({ "experts.ram", b });
    }
    const bool needs_disk = experts > ram;
    if (needs_disk && (!use_mmap || merged)) {
        ram_value.clear();
        R.notes.clear();
        off(!use_mmap ? "the experts do not fit in RAM, but reading them from disk needs mmap (drop --no-mmap)"
                      : "the experts do not fit in RAM, but merged up/gate expert tensors cannot be read from disk");
        return;
    }

    // VRAM per GPU: given, or auto: what is free less the GPU's weights, its KV cache and compute
    // buffers and a margin. The tier runs on the main GPU only for now.
    std::vector<int64_t> vram_req, margin_req;
    std::string err;
    if (!per_gpu("LA_EXPERTS_VRAM", n_gpus, main_gpu, vram_req, err)) throw std::runtime_error("--experts-vram: " + err);
    if (!per_gpu("LA_VRAM_MARGIN",  n_gpus, main_gpu, margin_req, err)) throw std::runtime_error("--experts-vram-margin: " + err);
    if (n_gpus == 0) {
        for (const auto & v : split_list(getenv("LA_EXPERTS_VRAM"))) {
            uint64_t b = 0;
            if (v != "auto" && llama_la_parse_size(v.c_str(), b) && b > 0) {
                LLAMA_LOG_WARN("expert cache: --experts-vram %s given, but no layer of the model is on a GPU; no VRAM tier\n", v.c_str());
            }
        }
    }
    int n_given = 0;
    for (int i = 0; i < n_gpus; ++i) {
        if (vram_req[i] <= 0) continue;
        ++n_given;
        if (i != main_gpu) {
            throw std::runtime_error("expert cache: a VRAM budget for " + gpus.name[i] + " is not supported yet: the VRAM "
                                     "tier runs on the main GPU (" + gpus.name[main_gpu] + ") only; give 0 for the others");
        }
    }
    if (n_given > 1) throw std::runtime_error("expert cache: VRAM budgets on more than one GPU are not supported yet");
    const bool have_profile = getenv("LA_PROFILE") && *getenv("LA_PROFILE");

    // the weights each GPU will hold (buffers that are not one GPU's, such as split ones, count for the main one)
    std::vector<uint64_t> gpu_weights(n_gpus, 0);
    for (const auto & bc : ctx_map) {
        if (n_gpus == 0 || !bc.first || !bc.second || ggml_backend_buft_is_host(bc.first)) continue;
        int g = main_gpu;
        for (int i = 0; i < n_gpus; ++i) if (gpus.buft[i] == bc.first) g = i;
        for (ggml_tensor * t = ggml_get_first_tensor(bc.second); t; t = ggml_get_next_tensor(bc.second, t)) gpu_weights[g] += ggml_nbytes(t);
    }
    // with LA_GPU_BATCH the tier also keeps scratch tensors as large as one layer's experts
    uint64_t transit = 0;
    if (getenv("LA_GPU_BATCH")) {
        for (const auto & l : model.layers) {
            uint64_t b = 0;
            for (const ggml_tensor * t : { l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps }) if (t) b += ggml_nbytes(t);
            transit = std::max(transit, b);
        }
    }

    const double min_frac = (double) model.hparams.n_expert_used / model.hparams.n_expert;   // one token's experts per layer
    std::vector<uint64_t> vram(n_gpus, 0), margin(n_gpus, 0);
    std::vector<std::string> vram_note(n_gpus);
    for (int i = 0; i < n_gpus; ++i) {
        margin[i] = margin_req[i] >= 0 ? (uint64_t) margin_req[i] : (gpus.display[i] ? 1ull << 30 : 512ull << 20);
        const uint64_t used = gpu_weights[i] + gpus.reserve[i] + margin[i] + transit;
        const uint64_t room = gpus.free[i] > used ? gpus.free[i] - used : 0;
        char b[384];
        snprintf(b, sizeof(b), "%.1f GiB free, less %.1f GiB of weights, %.1f GiB of KV cache and compute buffers%s and a %s margin%s",
                 gpus.free[i] / GiB, gpu_weights[i] / GiB, gpus.reserve[i] / GiB, transit ? ", scratch tensors for LA_GPU_BATCH" : "",
                 fmt_size(margin[i]).c_str(), gpus.display[i] ? " (it drives a display)" : "");
        if (vram_req[i] >= 0) {
            vram[i] = (uint64_t) vram_req[i];
            if (vram[i] > room) {
                LLAMA_LOG_WARN("expert cache: --experts-vram %s for %s is more than the %.1f GiB estimated to be left (%s)\n",
                               fmt_size(vram[i]).c_str(), gpus.name[i].c_str(), room / GiB, b);
            }
        } else if (i != main_gpu) {
            vram_note[i] = "auto: 0, the VRAM tier runs on the main GPU only for now";
        } else if (!gpus.offloading) {
            vram_note[i] = "auto: 0, no layer runs on the GPU (-ngl 0)";
        } else if (!have_profile) {
            vram_note[i] = "auto: 0, the VRAM tier needs a usage profile and no recipe gives one for this model";
        } else if (room < min_frac * experts) {
            vram_note[i] = std::string("auto: 0, less than one token's experts in every layer is left: ") + b;
        } else {
            vram[i] = std::min(room, experts);
            vram_note[i] = std::string("auto: ") + b;
        }
    }
    bool vram_given = false;
    for (int i = 0; i < n_gpus; ++i) vram_given = vram_given || vram_req[i] > 0;
    auto put_gpus = [&](bool on) {
        for (int i = 0; i < n_gpus; ++i) {
            R.gpus[i].kv = { { "vram", fmt_size(on ? vram[i] : 0) }, { "vram_margin", fmt_size(margin[i]) } };
            const std::string note = !on && vram_req[i] < 0 ? "auto: 0, the expert cache is off" : vram_note[i];
            if (!note.empty()) R.notes.push_back({ "devices." + gpus.name[i] + ".vram", note });
        }
    };

    char why[256];
    if (!needs_disk && !vram_given && mode != "on") {
        LLAMA_LOG_INFO("expert cache: off, not needed (%.1f GiB of experts, %.1f GiB of RAM for them%s)\n",
                       experts / GiB, ram / GiB, ram_auto ? ", auto" : "");
        setv("LA_ENABLE", "0");
        snprintf(why, sizeof(why), "auto: not needed, the %.1f GiB of experts fit the %.1f GiB of RAM for them", experts / GiB, ram / GiB);
        put_gpus(false);
        finish("off", why, "off, not needed");
        return;
    }

    double frac = needs_disk ? (double) ram / experts : 1.0;
    if (frac < min_frac) {
        LLAMA_LOG_WARN("expert cache: %.1f GiB of RAM is less than one token's experts in every layer; using %.1f GiB\n",
                       ram / GiB, min_frac * experts / GiB);
        frac = min_frac;
    }
    const uint64_t vram_main = n_gpus > 0 ? vram[main_gpu] : 0;
    const double gpu_frac = vram_main > 0 ? gpu_frac_for(model, ctx_map, vram_main, experts) : 0.0;

    char buf[64];
    setv("LA_ENABLE", "1");
    if (!h.real) setv("LA_REAL", needs_disk ? "1" : "0");
    if (!h.frac) { snprintf(buf, sizeof(buf), "%.6f", frac); setv("LA_RESIDENT_FRAC", buf); }
    if (!h.gpu && gpu_frac > 0) { snprintf(buf, sizeof(buf), "%.6f", gpu_frac); setv("LA_GPU_FRAC", buf); }
    if (needs_disk && ram_auto && !h.headroom) {
        snprintf(buf, sizeof(buf), "%" PRIu64, reserve / (1024 * 1024));
        setv("LA_HEADROOM_MB", buf);
    }
    LLAMA_LOG_INFO("expert cache: on, %.1f GiB of experts: %.1f GiB in RAM%s, %.1f GiB in VRAM%s, %s; quality %s\n",
                   experts / GiB, frac * experts / GiB, ram_auto ? " (auto)" : "",
                   (gpu_frac > 0 ? vram_main : 0) / GiB, n_gpus > 0 && vram_req[main_gpu] < 0 ? " (auto)" : "",
                   needs_disk ? "the rest read from disk" : "all in memory", level.c_str());
    if (mode == "on")    why[0] = 0;
    else if (needs_disk) snprintf(why, sizeof(why), "auto: the %.1f GiB of experts do not fit the %.1f GiB of RAM for them, the rest is read from disk", experts / GiB, ram / GiB);
    else                 snprintf(why, sizeof(why), "auto: a VRAM budget is given");
    put_gpus(true);
    finish("on", why, needs_disk ? "on, the experts that do not fit are read from disk" : "on, all experts in memory");
}
