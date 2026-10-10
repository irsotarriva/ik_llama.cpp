#pragma once

// Hot rows of a lookup table that stays on the file (--defer-ple: the per-layer token embedding).
//
// A token needs a few rows of the table, each a page of the file. Three things keep that cheap:
//   - the rows of a batch are known before any layer runs, so they are asked for at once on
//     LA_ROW_THREADS threads (default 8, 0 = off) instead of one after the other inside the graph;
//   - LA_ROW_PROFILE names a file with one count per page of the table: how often it was needed.
//     It is read at start and kept up to date, like LA_PROFILE for the experts;
//   - LA_ROW_RAM_MB: at start the most needed pages of that profile, up to this many MiB, are read
//     in the background, so the rows a conversation usually needs are in memory before it asks.
// The pages stay file-backed: when memory runs short the system drops them at no cost, so this
// tier only ever uses memory that is spare.

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

struct llama_la_rows {
    static constexpr size_t PAGE = 4096;

    const uint8_t * base      = nullptr;
    size_t          row_bytes = 0;
    int             threads   = 8;
    size_t          budget    = 0;      // pages read at start
    std::string     path;
    std::vector<uint8_t> count;         // per page, saturating
    std::thread          loader;
    std::atomic<bool>    stop{false};
    uint64_t tokens = 0, unsaved = 0;

    static llama_la_rows & get() {
        static llama_la_rows r;
        return r;
    }

    llama_la_rows() {
        if (const char * v = getenv("LA_ROW_THREADS")) threads = atoi(v);
        if (const char * v = getenv("LA_ROW_PROFILE")) path    = v;
        if (const char * v = getenv("LA_ROW_RAM_MB"))  budget  = (size_t) (std::max(0.0, atof(v)) * 1048576.0 / PAGE);
    }

    ~llama_la_rows() { detach(); }

    bool active() const { return threads > 0 || !path.empty(); }

    // touch `pages` of the table on several threads
    void fault_in(const std::vector<uint32_t> & pages, int n_threads) {
        std::atomic<size_t>   next{0};
        std::atomic<uint32_t> sink{0};
        auto work = [&] {
            uint32_t acc = 0;
            for (size_t i; (i = next.fetch_add(1)) < pages.size() && !stop.load(std::memory_order_relaxed); ) {
                acc += base[(size_t) pages[i] * PAGE];
            }
            sink.fetch_add(acc);
        };
        std::vector<std::thread> pool;
        for (int t = 1; t < n_threads; ++t) pool.emplace_back(work);
        work();
        for (auto & th : pool) th.join();
    }

    void attach(const ggml_tensor * table) {
        if (base == (const uint8_t *) table->data) return;
        detach();
        base      = (const uint8_t *) table->data;
        row_bytes = table->nb[1];
        stop      = false;
        if (path.empty()) return;
        const size_t n_pages = (ggml_nbytes(table) + PAGE - 1) / PAGE;
        count.assign(n_pages, 0);
        if (FILE * f = fopen(path.c_str(), "rb")) {
            std::vector<uint8_t> seen(n_pages);
            if (fread(seen.data(), 1, n_pages, f) == n_pages && fgetc(f) == EOF) count.swap(seen);
            fclose(f);
        }
        if (budget == 0) return;
        loader = std::thread([this] {
            std::vector<uint32_t> pages;
            for (size_t i = 0; i < count.size(); ++i) if (count[i]) pages.push_back((uint32_t) i);
            if (pages.empty()) return;
            if (pages.size() > budget) {
                std::nth_element(pages.begin(), pages.begin() + budget, pages.end(),
                                 [this](uint32_t a, uint32_t b) { return count[a] > count[b]; });
                pages.resize(budget);
            }
            std::sort(pages.begin(), pages.end());      // in file order
            const int64_t t0 = ggml_time_us();
            fault_in(pages, std::max(1, threads));
            if (!stop) fprintf(stderr, "la-rows: %zu pages of the lookup table (%.2f GiB) read ahead in %.1f s\n",
                               pages.size(), pages.size() * (double) PAGE / 1073741824.0, (ggml_time_us() - t0) / 1e6);
        });
    }

    // the rows of one batch, before the graph runs
    void touch(const ggml_tensor * table, const int32_t * rows, int64_t n_rows, int n_tokens) {
        if (!active() || !table || !table->data) return;
        attach(table);
        std::vector<uint32_t> pages;
        pages.reserve((size_t) n_rows * 2);
        for (int64_t r = 0; r < n_rows; ++r) {
            const size_t lo = (size_t) rows[r] * row_bytes, hi = lo + row_bytes - 1;
            pages.push_back((uint32_t) (lo / PAGE));
            if (hi / PAGE != lo / PAGE) pages.push_back((uint32_t) (hi / PAGE));
        }
        if (!count.empty()) {
            // the loader only reads the counts, and a count that is off by one changes nothing
            for (uint32_t p : pages) if (count[p] < 255) count[p]++;
            const uint64_t before = tokens;
            tokens  += (uint64_t) n_tokens;
            unsaved += (uint64_t) n_tokens;
            if ((before >> 16) != (tokens >> 16)) for (uint8_t & c : count) c >>= 1;   // forget slowly
            if (unsaved >= 1024) save();
        }
        if (threads > 0) fault_in(pages, (int) std::min<int64_t>(threads, std::max<int64_t>(1, n_rows / 16)));
    }

    void save() {
        unsaved = 0;
        if (path.empty() || count.empty()) return;
        const std::string tmp = path + ".tmp";
        FILE * f = fopen(tmp.c_str(), "wb");
        if (!f) return;
        const bool ok = fwrite(count.data(), 1, count.size(), f) == count.size();
        fclose(f);
        if (ok) {
            remove(path.c_str());
            rename(tmp.c_str(), path.c_str());
        }
    }

    // the model is going away: nothing may read the table after this
    void detach() {
        stop = true;
        if (loader.joinable()) loader.join();
        if (base && unsaved > 0) save();
        base = nullptr;
        count.clear();
    }
};
