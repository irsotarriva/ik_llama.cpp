#pragma once

// Client-controlled prompt cache on disk.
//
// A request can mark positions in its prompt (see "cache_breakpoints" in README-cache.md). The
// server takes a context checkpoint exactly there, and when the breakpoint asks for it the whole
// state of the sequence up to that position is written to LA_CACHE_DIR. A later request whose
// prompt starts with the same tokens - also after a restart of the server - loads that state
// instead of reading the tokens again.
//
// The client decides how long an entry lives: it is deleted when its time is up, and every use of
// it, or an explicit request to extend it, starts the time again. LA_CACHE_MAX_GB (default 20) is
// the server's own limit: over it, the entries that expire soonest go first.
//
// An entry is two files and a description:
//   <id>.state        the sequence (llama_state_seq_save_file), with its tokens
//   <id>.tokens.json  the tokens as the server keeps them
//   <id>.json         tokens, hash, model, expiry
// Lossless compression was measured and left out: zlib and lzma take 6% off a state of
// Qwen3.8-Flash-Next (8-bit KV cache plus recurrent state).
// A state is only valid for the model file and KV cache types that made it; both are in the key.

#include "server-common.h"

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

struct server_cache_entry {
    std::string id;
    std::string model;          // key of the model and cache layout
    int32_t  n_tokens  = 0;
    uint64_t hash      = 0;     // of the n_tokens tokens
    int64_t  ttl_s     = 0;
    int64_t  expires   = 0;     // unix seconds
    int64_t  created   = 0;
    int64_t  last_used = 0;
    int64_t  n_uses    = 0;
    int64_t  bytes     = 0;

    json to_json() const {
        return json{ {"id", id}, {"model", model}, {"n_tokens", n_tokens}, {"hash", hash}, {"ttl", ttl_s}, {"expires", expires},
                     {"created", created}, {"last_used", last_used}, {"n_uses", n_uses}, {"bytes", bytes} };
    }
    static server_cache_entry from_json(const json & j) {
        server_cache_entry e;
        e.id        = j.value("id", std::string());
        e.model     = j.value("model", std::string());
        e.n_tokens  = j.value("n_tokens", 0);
        e.hash      = j.value("hash", (uint64_t) 0);
        e.ttl_s     = j.value("ttl", (int64_t) 0);
        e.expires   = j.value("expires", (int64_t) 0);
        e.created   = j.value("created", (int64_t) 0);
        e.last_used = j.value("last_used", (int64_t) 0);
        e.n_uses    = j.value("n_uses", (int64_t) 0);
        e.bytes     = j.value("bytes", (int64_t) 0);
        return e;
    }
};

// "90", 90, "5m", "1h", "7d" -> seconds; 0 if it cannot be read
inline int64_t server_cache_parse_ttl(const json & v) {
    try {
        if (v.is_number()) {
            return std::max<int64_t>(0, v.get<int64_t>());
        }
        if (v.is_string()) {
            const std::string s = v.get<std::string>();
            if (s.empty()) {
                return 0;
            }
            size_t used = 0;
            const double x = std::stod(s, &used);
            const char unit = used < s.size() ? s[used] : 's';
            const double mul = unit == 'm' ? 60 : unit == 'h' ? 3600 : unit == 'd' ? 86400 : 1;
            return std::max<int64_t>(0, (int64_t) (x * mul));
        }
    } catch (...) {
    }
    return 0;
}

struct server_cache_store {
    std::string dir;
    std::string model_key;
    int64_t     max_bytes = 0;
    std::map<std::string, server_cache_entry> entries;
    mutable std::mutex mtx;

    static int64_t now() { return (int64_t) time(nullptr); }

    bool enabled() const { return !dir.empty(); }

    static uint64_t hash_step(uint64_t h, llama_token t) {
        for (int b = 0; b < 4; ++b) {
            h ^= (uint64_t) ((uint32_t) t >> (8 * b)) & 0xffu;
            h *= 1099511628211ull;
        }
        return h;
    }
    static constexpr uint64_t HASH_START = 1469598103934665603ull;

    std::string path(const std::string & id, const char * ext) const { return dir + "/" + id + ext; }

    // LA_CACHE_DIR turns the store on. key: what a state depends on besides its tokens.
    void init(const std::string & key) {
        std::lock_guard<std::mutex> lock(mtx);
        model_key = key;
        const char * d = getenv("LA_CACHE_DIR");
        if (!d || !*d) {
            return;
        }
        dir = d;
        const char * m = getenv("LA_CACHE_MAX_GB");
        max_bytes = (int64_t) ((m ? atof(m) : 20.0) * 1024.0 * 1024.0 * 1024.0);
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        for (const auto & f : std::filesystem::directory_iterator(dir, ec)) {
            if (f.path().extension() != ".json" || f.path().string().find(".tokens.json") != std::string::npos) {
                continue;
            }
            try {
                std::ifstream in(f.path());
                server_cache_entry e = server_cache_entry::from_json(json::parse(in));
                if (!e.id.empty()) {
                    entries[e.id] = e;
                }
            } catch (...) {
            }
        }
        purge_locked();
        LLAMA_LOG_INFO("prompt cache store: %s, %zu entries, limit %.1f GiB\n", dir.c_str(), entries.size(), max_bytes / 1073741824.0);
    }

    void write_meta(const server_cache_entry & e) const {
        const std::string tmp = path(e.id, ".json.tmp");
        {
            std::ofstream out(tmp);
            out << e.to_json().dump();
        }
        std::error_code ec;
        std::filesystem::rename(tmp, path(e.id, ".json"), ec);
    }

    void remove_locked(const std::string & id) {
        std::error_code ec;
        for (const char * ext : { ".state", ".tokens.json", ".json" }) {
            std::filesystem::remove(path(id, ext), ec);
        }
        entries.erase(id);
    }

    // drop what has expired, then the entries that expire soonest while over the size limit
    void purge_locked() {
        const int64_t t = now();
        std::vector<std::string> gone;
        int64_t total = 0;
        for (const auto & kv : entries) {
            if (kv.second.expires <= t) {
                gone.push_back(kv.first);
            } else {
                total += kv.second.bytes;
            }
        }
        for (const auto & id : gone) {
            LLAMA_LOG_INFO("prompt cache store: %s expired\n", id.c_str());
            remove_locked(id);
        }
        while (max_bytes > 0 && total > max_bytes && !entries.empty()) {
            auto soonest = entries.begin();
            for (auto it = entries.begin(); it != entries.end(); ++it) {
                if (it->second.expires < soonest->second.expires) {
                    soonest = it;
                }
            }
            LLAMA_LOG_INFO("prompt cache store: over the size limit, dropping %s\n", soonest->first.c_str());
            total -= soonest->second.bytes;
            remove_locked(soonest->first);
        }
    }

    // The longest entry whose tokens start the prompt. Using an entry starts its time again.
    bool find(const std::vector<llama_token> & prompt, server_cache_entry & out) {
        std::lock_guard<std::mutex> lock(mtx);
        if (!enabled()) {
            return false;
        }
        purge_locked();
        std::map<int32_t, uint64_t> at;     // prefix length -> hash, for the lengths that have entries
        for (const auto & kv : entries) {
            if (kv.second.model == model_key && kv.second.n_tokens <= (int32_t) prompt.size()) {
                at[kv.second.n_tokens] = 0;
            }
        }
        if (at.empty()) {
            return false;
        }
        uint64_t h = HASH_START;
        int32_t  i = 0;
        for (auto & len : at) {
            for (; i < len.first; ++i) {
                h = hash_step(h, prompt[i]);
            }
            len.second = h;
        }
        server_cache_entry * best = nullptr;
        for (auto & kv : entries) {
            server_cache_entry & e = kv.second;
            if (e.model != model_key || e.n_tokens > (int32_t) prompt.size() || at[e.n_tokens] != e.hash) {
                continue;
            }
            touch_locked(e);
            if (!best || e.n_tokens > best->n_tokens) {
                best = &e;
            }
        }
        if (best) {
            out = *best;
        }
        return best != nullptr;
    }

    void touch_locked(server_cache_entry & e) {
        e.last_used = now();
        e.n_uses++;
        e.expires = e.last_used + e.ttl_s;
        write_meta(e);
    }

    // id of the entry for these tokens (it need not exist)
    // kind: empty for the state of a prompt, "tab" for a tab. It is part of the id and of the entry's
    // model key, so a tab is never loaded as the start of a prompt with the same tokens.
    std::string id_for(const std::vector<llama_token> & tokens, uint64_t & hash, const std::string & kind = "") const {
        hash = HASH_START;
        for (llama_token t : tokens) {
            hash = hash_step(hash, t);
        }
        uint64_t k = hash;
        for (char c : model_key + kind) {
            k ^= (uint8_t) c;
            k *= 1099511628211ull;
        }
        char buf[64];
        snprintf(buf, sizeof(buf), "%016llx-%d", (unsigned long long) k, (int) tokens.size());
        return buf;
    }

    bool has(const std::string & id) {
        std::lock_guard<std::mutex> lock(mtx);
        return entries.count(id) > 0;
    }

    // the entry with this id, if it is of this model and kind. Using it starts its time again.
    bool lookup(const std::string & id, const std::string & kind, server_cache_entry & out) {
        std::lock_guard<std::mutex> lock(mtx);
        if (!enabled()) {
            return false;
        }
        purge_locked();
        auto it = entries.find(id);
        if (it == entries.end() || it->second.model != model_key + kind) {
            return false;
        }
        touch_locked(it->second);
        out = it->second;
        return true;
    }

    // the files of a new entry are written; record it
    void add(server_cache_entry e, const std::string & kind = "") {
        std::lock_guard<std::mutex> lock(mtx);
        e.model   = model_key + kind;
        e.created = e.last_used = now();
        e.expires = e.created + e.ttl_s;
        e.bytes   = 0;
        std::error_code ec;
        for (const char * ext : { ".state", ".tokens.json" }) {
            const auto n = std::filesystem::file_size(path(e.id, ext), ec);
            if (!ec) {
                e.bytes += (int64_t) n;
            }
        }
        entries[e.id] = e;
        write_meta(e);
        purge_locked();
    }

    // an entry gets a new lifetime (ttl_s > 0) or only its time started again (ttl_s == 0)
    bool extend(const std::string & id, int64_t ttl_s, server_cache_entry & out) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = entries.find(id);
        if (it == entries.end()) {
            return false;
        }
        if (ttl_s > 0) {
            it->second.ttl_s = ttl_s;
        }
        it->second.expires = now() + it->second.ttl_s;
        write_meta(it->second);
        out = it->second;
        return true;
    }

    bool remove(const std::string & id) {
        std::lock_guard<std::mutex> lock(mtx);
        if (!entries.count(id)) {
            return false;
        }
        remove_locked(id);
        return true;
    }

    json list() {
        std::lock_guard<std::mutex> lock(mtx);
        purge_locked();
        json out = json::array();
        const int64_t t = now();
        for (const auto & kv : entries) {
            json j = kv.second.to_json();
            j["expires_in"] = kv.second.expires - t;
            j["usable"]     = kv.second.model == model_key || kv.second.model == model_key + "|tab";
            j["kind"]       = kv.second.model.size() > 4 && kv.second.model.compare(kv.second.model.size() - 4, 4, "|tab") == 0 ? "tab" : "prompt";
            out.push_back(j);
        }
        return json{ {"enabled", enabled()}, {"dir", dir}, {"limit_bytes", max_bytes}, {"entries", out} };
    }
};
