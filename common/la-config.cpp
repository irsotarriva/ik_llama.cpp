#include "la-config.h"

#include "llama.h"

#include <nlohmann/json.hpp>
#define TOML_EXCEPTIONS 1
#include <tomlplusplus/toml.hpp>

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

bool common_la_size_ok(const std::string & s) {
    static const std::regex re("^ *[0-9]*\\.?[0-9]+ *([gGmMtT]([iI]?[bB])?)? *$");
    return std::regex_match(s, re);
}

namespace {

std::string lower(std::string s) {
    for (char & c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

void set_env(const char * name, const std::string & value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

bool env_set(const char * name) {
    const char * v = getenv(name);
    return v != nullptr && *v != 0;
}

// size in GiB, for LA_CACHE_MAX_GB
double size_gib(const std::string & v) {
    double gb = std::stod(v);
    const size_t at = v.find_first_not_of("0123456789. ");
    const char u = at == std::string::npos ? 'g' : (char) std::tolower((unsigned char) v[at]);
    if (u == 'm') gb /= 1024.0;
    if (u == 't') gb *= 1024.0;
    return gb;
}

json from_toml(const toml::node & n) {
    if (const auto * t = n.as_table()) {
        json o = json::object();
        for (const auto & [k, v] : *t) o[std::string(k.str())] = from_toml(v);
        return o;
    }
    if (const auto * a = n.as_array()) {
        json o = json::array();
        for (const auto & v : *a) o.push_back(from_toml(v));
        return o;
    }
    if (const auto * s = n.as_string())         return s->get();
    if (const auto * i = n.as_integer())        return i->get();
    if (const auto * f = n.as_floating_point()) return f->get();
    if (const auto * b = n.as_boolean())        return b->get();
    std::ostringstream ss;                               // dates and times: kept as text
    if (const auto * d = n.as_date())           ss << *d;
    else if (const auto * t = n.as_time())      ss << *t;
    else if (const auto * dt = n.as_date_time()) ss << *dt;
    return ss.str();
}

struct reader {
    std::string & err;

    bool fail(const std::string & key, const std::string & what) {
        err = key + ": " + what;
        return false;
    }

    // a string from a fixed set
    bool choice(const json & j, const std::string & key, std::initializer_list<const char *> allowed, std::optional<std::string> & out) {
        std::string list;
        for (const char * a : allowed) list += (list.empty() ? "" : ", ") + std::string(a);
        if (!j.is_string()) return fail(key, "takes one of " + list);
        const std::string v = lower(j.get<std::string>());
        for (const char * a : allowed) {
            if (v == a) { out = v; return true; }
        }
        return fail(key, "takes one of " + list + ", not '" + j.get<std::string>() + "'");
    }

    // "24G", 24 (GiB), or "auto" where allowed
    bool size(const json & j, const std::string & key, bool allow_auto, std::optional<std::string> & out) {
        const std::string expect = std::string("takes a size such as \"24G\" or \"512M\"") + (allow_auto ? ", or \"auto\"" : "");
        if (j.is_number()) {
            const double v = j.get<double>();
            if (!(v >= 0) || !std::isfinite(v)) return fail(key, expect);
            char buf[64];
            snprintf(buf, sizeof(buf), "%gG", v);
            out = buf;
            return true;
        }
        if (!j.is_string()) return fail(key, expect);
        const std::string v = lower(j.get<std::string>());
        if (allow_auto && v == "auto") { out = v; return true; }
        if (!common_la_size_ok(v)) return fail(key, expect + ", not '" + j.get<std::string>() + "'");
        out = j.get<std::string>();
        return true;
    }

    bool table(const json & j, const std::string & key) {
        return j.is_object() || fail(key, "should be a table ([" + key + "])");
    }

    bool unknown(const std::string & key) {
        return fail(key, "unknown key");
    }
};

// keys kept for later versions, so that a file written for one says so instead of "unknown key"
bool reserved(const std::string & key, std::string & err) {
    err = key + ": not supported yet by this version";
    return false;
}

} // namespace

bool common_la_config_from_json(const json & j, common_la_config & cfg, std::string & err) {
    reader r{ err };
    if (!j.is_object()) return r.fail("config", "should be a table");
    for (const auto & [section, body] : j.items()) {
        if (section == "notes" || section == "summary") continue;          // written by --print-config
        if (section == "version") {
            if (!body.is_number_integer() || body.get<int64_t>() != 1) return r.fail("version", "this engine reads version 1");
            continue;
        }
        if (section == "experts") {
            if (!r.table(body, section)) return false;
            for (const auto & [k, v] : body.items()) {
                const std::string key = section + "." + k;
                if      (k == "cache")   { if (!r.choice(v, key, { "auto", "on", "off" }, cfg.cache)) return false; }
                else if (k == "quality") { if (!r.choice(v, key, { "exact", "near", "balanced", "fast" }, cfg.quality)) return false; }
                else if (k == "ram")     { if (!r.size(v, key, true, cfg.ram)) return false; }
                else if (k == "layers")  { return reserved(key, err); }
                else return r.unknown(key);
            }
        } else if (section == "devices") {
            if (!r.table(body, section)) return false;
            for (const auto & [name, dev] : body.items()) {
                const std::string base = section + "." + name;
                if (!r.table(dev, base)) return false;
                if (name.empty() || !std::isdigit((unsigned char) name.back())) {
                    return r.fail(base, "a device is named as the backend names it, such as CUDA0 (the number is its place)");
                }
                auto & d = cfg.devices[name];
                for (const auto & [k, v] : dev.items()) {
                    const std::string key = base + "." + k;
                    if      (k == "vram")        { if (!r.size(v, key, true, d.vram)) return false; }
                    else if (k == "vram_margin") { if (!r.size(v, key, false, d.vram_margin)) return false; }
                    else if (k == "layers")      { return reserved(key, err); }
                    else return r.unknown(key);
                }
            }
        } else if (section == "cache_disk") {
            if (!r.table(body, section)) return false;
            for (const auto & [k, v] : body.items()) {
                const std::string key = section + "." + k;
                if (k == "dir") {
                    if (!v.is_string()) return r.fail(key, "takes a folder");
                    cfg.cache_disk_dir = v.get<std::string>();
                } else if (k == "max") {
                    if (!r.size(v, key, false, cfg.cache_disk_max)) return false;
                } else {
                    return r.unknown(key);
                }
            }
        } else if (section == "layers") {
            return reserved(section, err);
        } else {
            return r.unknown(section);
        }
    }
    return true;
}

json common_la_config_to_json(const common_la_config & cfg) {
    json j = json::object();
    json experts = json::object();
    if (cfg.cache)   experts["cache"]   = *cfg.cache;
    if (cfg.quality) experts["quality"] = *cfg.quality;
    if (cfg.ram)     experts["ram"]     = *cfg.ram;
    if (!experts.empty()) j["experts"] = experts;
    json devices = json::object();
    for (const auto & [name, d] : cfg.devices) {
        json o = json::object();
        if (d.vram)        o["vram"]        = *d.vram;
        if (d.vram_margin) o["vram_margin"] = *d.vram_margin;
        devices[name] = o;
    }
    if (!devices.empty()) j["devices"] = devices;
    json disk = json::object();
    if (cfg.cache_disk_dir) disk["dir"] = *cfg.cache_disk_dir;
    if (cfg.cache_disk_max) disk["max"] = *cfg.cache_disk_max;
    if (!disk.empty()) j["cache_disk"] = disk;
    return j;
}

bool common_la_config_load(const std::string & path, common_la_config & cfg, std::string & err) {
    json j;
    const bool is_json = lower(fs::path(path).extension().string()) == ".json";
    try {
        if (is_json) {
            std::ifstream f(path);
            if (!f) { err = path + ": cannot open"; return false; }
            j = json::parse(f);
        } else {
            std::ifstream f(path, std::ios::binary);
            if (!f) { err = path + ": cannot open"; return false; }
            std::stringstream ss;
            ss << f.rdbuf();
            j = from_toml(toml::parse(ss.str(), path));
        }
    } catch (const toml::parse_error & e) {
        std::ostringstream ss;
        ss << path << ":" << e.source().begin.line << ":" << e.source().begin.column << ": " << e.description();
        err = ss.str();
        return false;
    } catch (const std::exception & e) {
        err = path + ": " + e.what();
        return false;
    }
    std::string why;
    if (!common_la_config_from_json(j, cfg, why)) {
        err = path + ": " + why;
        return false;
    }
    if (cfg.cache_disk_dir && !cfg.cache_disk_dir->empty() && fs::path(*cfg.cache_disk_dir).is_relative()) {
        cfg.cache_disk_dir = (fs::absolute(fs::path(path)).parent_path() / *cfg.cache_disk_dir).string();
    }
    return true;
}

void common_la_config_apply(const common_la_config & cfg) {
    auto put = [](const char * name, const std::optional<std::string> & v) {
        if (v && !env_set(name)) set_env(name, *v);
    };
    put("LA_EXPERT_CACHE", cfg.cache);
    put("LA_RECIPE_LEVEL", cfg.quality);
    put("LA_EXPERTS_RAM",  cfg.ram);
    // by name, so that the engine places them by device number: "CUDA0=6G,CUDA1=auto"
    std::string vram, margin;
    for (const auto & [name, d] : cfg.devices) {
        if (d.vram)        vram   += (vram.empty()   ? "" : ",") + name + "=" + *d.vram;
        if (d.vram_margin) margin += (margin.empty() ? "" : ",") + name + "=" + *d.vram_margin;
    }
    if (!vram.empty())   put("LA_EXPERTS_VRAM", vram);
    if (!margin.empty()) put("LA_VRAM_MARGIN",  margin);
    put("LA_CACHE_DIR", cfg.cache_disk_dir);
    if (cfg.cache_disk_max) put("LA_CACHE_MAX_GB", std::to_string(size_gib(*cfg.cache_disk_max)));
}

json common_la_config_resolved_json() {
    json j = json::object();
    const char * s = llama_expert_cache_config();
    if (s && *s) {
        try { j = json::parse(s); } catch (const std::exception &) { j = json::object(); }
    }
    // the disk prompt cache is the server's: its settings are as given
    json disk = json::object();
    const char * dir = getenv("LA_CACHE_DIR");
    const char * max = getenv("LA_CACHE_MAX_GB");
    if (dir && *dir) disk["dir"] = dir;
    char buf[64];
    snprintf(buf, sizeof(buf), "%gG", max && *max ? atof(max) : 20.0);
    disk["max"] = buf;
    j["cache_disk"] = disk;
    if (!j.contains("notes")) j["notes"] = json::object();
    if (!(dir && *dir)) j["notes"]["cache_disk.dir"] = "none: the server keeps no prompt states on disk";
    return j;
}

std::string common_la_config_resolved_toml(const std::string & model_path) {
    const json j = common_la_config_resolved_json();
    const json notes = j.value("notes", json::object());
    std::ostringstream o;
    auto note = [&](const std::string & key) {
        if (notes.contains(key) && notes[key].is_string()) o << "# " << notes[key].get<std::string>() << "\n";
    };
    auto value = [&](const std::string & k, const json & v) {
        // a JSON string is a valid TOML basic string
        o << k << " = " << (v.is_string() ? v.dump() : json(v.dump()).dump()) << "\n";
    };
    o << "# ik_llama.cpp config resolved for " << fs::path(model_path).filename().string();
    if (j.contains("summary")) o << ": expert cache " << j["summary"].get<std::string>();
    o << "\n# Every automatic choice is filled in; the comments say how it was made. Use with --config.\n";
    o << "version = 1\n";
    if (j.contains("experts")) {
        o << "\n[experts]\n";
        for (const auto & [k, v] : j["experts"].items()) { note("experts." + k); value(k, v); }
    }
    if (j.contains("devices")) {
        for (const auto & [name, d] : j["devices"].items()) {
            o << "\n[devices." << name << "]";
            if (notes.contains("devices." + name)) o << "   # " << notes["devices." + name].get<std::string>();
            o << "\n";
            for (const auto & [k, v] : d.items()) { note("devices." + name + "." + k); value(k, v); }
        }
    }
    o << "\n[cache_disk]\n";
    const json & disk = j["cache_disk"];
    note("cache_disk.dir");
    if (disk.contains("dir")) value("dir", disk["dir"]); else o << "# dir = \"prompt-cache\"\n";
    value("max", disk["max"]);
    return o.str();
}
