#include "llama-la-recipe.h"
#include "llama-model.h"
#include "llama-arch.h"
#include "llama-impl.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace {

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

void llama_la_apply_recipe(const llama_model & model, const std::string & fname) {
    static std::atomic<bool> done{false};
    const char * where = getenv("LA_RECIPES");
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

    std::string level = best->default_level;
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
