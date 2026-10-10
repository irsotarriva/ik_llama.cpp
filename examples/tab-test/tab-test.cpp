// Tabbed context on the engine's sequence mechanism: a check that it is exact.
//
// A tab is a sequence of its own with positions from 0. Its cells also carry the id of the agent
// sequence, so a tab token attends only to its own tab (its first id) and an agent token, placed
// after the longest tab, attends to every tab and to the agent tokens before it. That is the tab
// mask of I:\recursiveLocal\docs\TABBED_CONTEXT.md. Nothing is trained here: the question is only
// whether tabs encoded apart, in any order, and replaced one at a time give the agent the same
// logits as the whole layout computed in one pass.
//
//   llama-tab-test -m model.gguf [-t threads]
//
// Needs a model with attention only. A model with recurrent state per sequence cannot see the tabs
// through that state.

#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

struct layout {
    std::vector<std::vector<llama_token>> tabs;
    std::vector<llama_token> agent;
    llama_pos agent_pos() const {
        size_t n = 0;
        for (const auto & t : tabs) n = std::max(n, t.size());
        return (llama_pos) n;
    }
};

static bool decode(llama_context * ctx, llama_batch & batch) {
    if (batch.n_tokens == 0) return true;
    const int rc = llama_decode(ctx, batch);
    if (rc != 0) fprintf(stderr, "llama_decode returned %d\n", rc);
    common_batch_clear(batch);
    return rc == 0;
}

static void add_tab(llama_batch & batch, const std::vector<llama_token> & tab, llama_seq_id id, llama_seq_id agent) {
    for (size_t i = 0; i < tab.size(); ++i) common_batch_add(batch, tab[i], (llama_pos) i, { id, agent }, false);
}

static void add_agent(llama_batch & batch, const layout & L, llama_seq_id agent) {
    for (size_t i = 0; i < L.agent.size(); ++i) {
        common_batch_add(batch, L.agent[i], L.agent_pos() + (llama_pos) i, { agent }, i + 1 == L.agent.size());
    }
}

// the logits of the last agent token
static std::vector<float> read_logits(llama_context * ctx, const llama_model * model, int i_batch) {
    const float * p = llama_get_logits_ith(ctx, i_batch);
    return std::vector<float>(p, p + llama_n_vocab(model));
}

static double max_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double d = 0;
    for (size_t i = 0; i < a.size(); ++i) d = std::max(d, (double) std::fabs(a[i] - b[i]));
    return d;
}

static int top(const std::vector<float> & a) { return (int) (std::max_element(a.begin(), a.end()) - a.begin()); }

int main(int argc, char ** argv) {
    gpt_params params;
    if (!gpt_params_parse(argc, argv, params)) return 1;

    llama_backend_init();
    llama_model * model = llama_model_load_from_file(params.model.c_str(), common_model_params_to_llama(params));
    if (!model) return 1;
    llama_context_params cp = common_context_params_to_llama(params);
    cp.n_ctx     = 4096;
    cp.n_batch   = 2048;
    cp.n_ubatch  = 2048;
    cp.n_seq_max = 8;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;

    const std::vector<std::string> texts = {
        "File config.py\nMAX_RETRIES = 7\nTIMEOUT_SECONDS = 30\nDEFAULT_REGION = \"north\"\n",
        "File users.py\nimport config\n\ndef greeting(name):\n    return \"Good morning, \" + name + \" from the \" + config.DEFAULT_REGION\n",
        "File notes.txt\nThe harbour office opens at nine. The ferry to the island leaves at a quarter past ten and takes forty minutes.\n",
    };
    const std::string edited = "File config.py\nMAX_RETRIES = 3\nTIMEOUT_SECONDS = 120\nDEFAULT_REGION = \"south\"\nVERBOSE = True\nLOG_FILE = \"run.log\"\n";
    const std::string query  = "Question: what is the value of DEFAULT_REGION in config.py?\nAnswer: DEFAULT_REGION is \"";

    layout L;
    for (const auto & t : texts) L.tabs.push_back(common_tokenize(model, t, true));
    L.agent = common_tokenize(model, query, false);
    const llama_seq_id A = 7;
    llama_batch batch = llama_batch_init(2048, 0, 2);
    bool ok = true;

    // 1. the whole layout in one pass
    llama_kv_cache_clear(ctx);
    for (size_t t = 0; t < L.tabs.size(); ++t) add_tab(batch, L.tabs[t], (llama_seq_id) t, A);
    add_agent(batch, L, A);
    int i_last = batch.n_tokens - 1;
    ok = decode(ctx, batch) && ok;
    const std::vector<float> one_pass = read_logits(ctx, model, i_last);

    // the same again, and with the tabs in another order inside the one batch: how much does a rerun move?
    auto whole = [&](const std::vector<int> & order) {
        llama_kv_cache_clear(ctx);
        for (int t : order) add_tab(batch, L.tabs[t], (llama_seq_id) t, A);
        add_agent(batch, L, A);
        const int i = batch.n_tokens - 1;
        ok = decode(ctx, batch) && ok;
        return read_logits(ctx, model, i);
    };
    const std::vector<float> one_pass_again = whole({ 0, 1, 2 });
    const std::vector<float> one_pass_other = whole({ 2, 0, 1 });

    // 2. every tab in a call of its own, then the agent
    auto apart =[&](const layout & lay, const std::vector<int> & order) {
        llama_kv_cache_clear(ctx);
        for (int t : order) {
            add_tab(batch, lay.tabs[t], (llama_seq_id) t, A);
            ok = decode(ctx, batch) && ok;
        }
        add_agent(batch, lay, A);
        const int i = batch.n_tokens - 1;
        ok = decode(ctx, batch) && ok;
        return read_logits(ctx, model, i);
    };
    const std::vector<float> in_order  = apart(L, { 0, 1, 2 });
    const std::vector<float> reordered = apart(L, { 2, 0, 1 });

    // 3. edit tab 0 in place: drop the agent's tokens and the tab's cells, encode the new tab, ask again
    layout E = L;
    E.tabs[0] = common_tokenize(model, edited, true);
    llama_kv_cache_seq_rm(ctx, A, L.agent_pos(), -1);
    const int n_dropped = llama_kv_cache_seq_rm_cells(ctx, 0);
    add_tab(batch, E.tabs[0], 0, A);
    ok = decode(ctx, batch) && ok;
    add_agent(batch, E, A);
    i_last = batch.n_tokens - 1;
    ok = decode(ctx, batch) && ok;
    const std::vector<float> after_edit = read_logits(ctx, model, i_last);
    const std::vector<float> edited_fresh = apart(E, { 0, 1, 2 });

    // 4. for scale: the same text as one ordinary sequence
    llama_kv_cache_clear(ctx);
    {
        llama_pos pos = 0;
        for (const auto & tab : L.tabs) for (llama_token tok : tab) common_batch_add(batch, tok, pos++, { 0 }, false);
        for (size_t i = 0; i < L.agent.size(); ++i) common_batch_add(batch, L.agent[i], pos++, { 0 }, i + 1 == L.agent.size());
    }
    i_last = batch.n_tokens - 1;
    ok = decode(ctx, batch) && ok;
    const std::vector<float> causal = read_logits(ctx, model, i_last);

    // the yardstick: the same ordinary sequence decoded in pieces of the sizes the tabs have. The engine's
    // kernels do not give bit-equal results for different batch sizes, and tabs encoded apart are
    // different batches by construction.
    llama_kv_cache_clear(ctx);
    {
        llama_pos pos = 0;
        for (const auto & tab : L.tabs) {
            for (llama_token tok : tab) common_batch_add(batch, tok, pos++, { 0 }, false);
            ok = decode(ctx, batch) && ok;
        }
        for (size_t i = 0; i < L.agent.size(); ++i) common_batch_add(batch, L.agent[i], pos++, { 0 }, i + 1 == L.agent.size());
    }
    i_last = batch.n_tokens - 1;
    ok = decode(ctx, batch) && ok;
    const std::vector<float> causal_pieces = read_logits(ctx, model, i_last);
    const double noise = max_diff(causal, causal_pieces);

    auto word = [&](const std::vector<float> & l) { return common_token_to_piece(ctx, top(l)); };
    const double range = *std::max_element(one_pass.begin(), one_pass.end()) - *std::min_element(one_pass.begin(), one_pass.end());
    printf("\ntabs of %zu, %zu and %zu tokens, agent of %zu tokens at position %d; logits span %.1f\n",
           L.tabs[0].size(), L.tabs[1].size(), L.tabs[2].size(), L.agent.size(), (int) L.agent_pos(), range);
    printf("%-58s %12s  %s\n", "", "max |dlogit|", "next token");
    printf("%-58s %12s  '%s'\n", "whole layout in one pass (the reference)", "", word(one_pass).c_str());
    printf("%-58s %12.2e  '%s'\n", "the same run again", max_diff(one_pass, one_pass_again), word(one_pass_again).c_str());
    printf("%-58s %12.2e  '%s'\n", "one pass, tabs in another order in the batch", max_diff(one_pass, one_pass_other), word(one_pass_other).c_str());
    printf("%-58s %12.2e  '%s'\n", "tabs encoded apart, in order", max_diff(one_pass, in_order), word(in_order).c_str());
    printf("%-58s %12.2e  '%s'\n", "tabs encoded apart, other order", max_diff(one_pass, reordered), word(reordered).c_str());
    printf("%-58s %12.2e  '%s'\n", "tab 0 edited in place, against a fresh build of the edit", max_diff(after_edit, edited_fresh), word(after_edit).c_str());
    printf("%-58s %12.2e  '%s'\n", "what the edit changed (should be large)", max_diff(one_pass, after_edit), word(after_edit).c_str());
    printf("%-58s %12.2e  '%s'\n", "one ordinary sequence, for scale", max_diff(one_pass, causal), word(causal).c_str());
    printf("%-58s %12.2e  '%s'\n", "ordinary sequence in one batch against in pieces", noise, word(causal_pieces).c_str());
    printf("the edit dropped %d cells (tab 0 had %zu tokens)\n", n_dropped, L.tabs[0].size());

    // Exact here means: no further from the reference than the engine is from itself when an ordinary
    // sequence is decoded in batches of other sizes, and far closer than a real change of a tab.
    const double tol = 1.5 * noise;
    const double worst = std::max({ max_diff(one_pass, in_order), max_diff(one_pass, reordered), max_diff(after_edit, edited_fresh) });
    const bool pass = ok && worst <= tol && n_dropped == (int) L.tabs[0].size() && max_diff(one_pass, after_edit) > 5 * tol;
    printf("\n%s: tabs encoded apart are within %.2f of the layout in one pass; the engine's own batch-size noise is %.2f, an edit moves %.2f\n",
           pass ? "PASS" : "FAIL", worst, noise, max_diff(one_pass, after_edit));

    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return pass ? 0 : 1;
}
