#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

// The split tab model (I:\recursiveLocal, runs\tabs_engine).
//
// The first n_tab_split layers are an encoder: ordinary causal layers with their own K/V in which a
// segment (one tab, or the agent stream) sees only itself. After them every token gets one shared
// K/V pair. The remaining layers are a decoder: each makes only a query and reads the shared K/V,
// where an agent token also sees every tab.
//
// What follows for the engine:
//   - the shared K/V lives in the cache slot of layer n_tab_split; the decoder layers have none
//   - nothing reads the decoder state of another token, so the decoder runs only on the tokens
//     whose logits were asked for. A tab is encoder work only, and so is a prompt but its last token
//   - a tab is a sequence of its own. The two masks are inputs of their own (llama_set_inputs): in
//     the encoder a token sees the earlier cells of its sequence, in the decoder an agent token also
//     sees every cell of the tabs attached to its sequence (llama_tab_attach)
//   - an agent stream's positions are given from 0; n_tab_agent_pos is added where they are set

ggml_cgraph * llm_build_context::build_tabsplit() {
    ggml_cgraph * gf = new_graph_custom();

    GGML_ASSERT(batch.token && "the split tab model takes tokens, not embeddings");

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    const int     n_split     = hparams.n_tab_split;
    const float   kq_scale    = 1.0f/sqrtf(float(n_embd_head));
    const int64_t n_ff_exp    = hparams.n_ff_exp;

    auto new_mask = [&](ggml_tensor *& slot, int64_t n_rows, const char * name) {
        slot = ggml_new_tensor_2d(ctx0, flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32, n_kv, GGML_PAD(n_rows, GGML_KQ_MASK_PAD));
        cb(slot, name, -1);
        ggml_set_input(slot);
        return slot;
    };

    auto rope = [&](ggml_tensor * x, ggml_tensor * pos, int64_t n_heads, int64_t n_rows) {
        x = ggml_reshape_3d(ctx0, x, n_embd_head, n_heads, n_rows);
        return ggml_rope_ext(ctx0, x, pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
    };

    // the always-on expert plus the top routed experts, mixed by a softmax over the chosen scores
    auto moe = [&](ggml_tensor * inp, int il, int64_t n_rows) {
        const auto & layer = model.layers[il];
        ggml_tensor * x = llm_build_norm(ctx0, inp, hparams, layer.ffn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(x, "ffn_norm", il);

        ggml_tensor * shared = llm_build_ffn(ctx0, lctx, nullptr, x,
                layer.ffn_up,   layer.ffn_up_b,   NULL,
                NULL,           NULL,             NULL,
                layer.ffn_down, layer.ffn_down_b, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_SEQ, cb, il);
        cb(shared, "ffn_shared", il);

        ggml_tensor * logits = llm_build_lora_mm(lctx, ctx0, layer.ffn_gate_inp, x);         // [n_expert, n_rows]
        cb(logits, "ffn_moe_logits", il);
        ggml_tensor * selected = ggml_top_k(ctx0, logits, n_expert_used);                    // [n_expert_used, n_rows]
        cb(selected, "ffn_moe_topk", il);
        ggml_tensor * weights = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, logits, 1, n_expert, n_rows), selected);
        weights = ggml_soft_max(ctx0, ggml_reshape_2d(ctx0, weights, n_expert_used, n_rows));
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_rows);
        cb(weights, "ffn_moe_weights", il);
        ggml_build_forward_expand(gf, weights);

        ggml_tensor * x3 = ggml_reshape_3d(ctx0, x, n_embd, 1, n_rows);
        ggml_tensor * h  = ggml_mul_mat_id(ctx0, layer.ffn_up_exps, x3, selected);           // [n_ff_exp, n_expert_used, n_rows]
        h = ggml_add_id(ctx0, h, layer.ffn_up_exps_b, selected);
        h = ggml_silu(ctx0, h);
        cb(h, "ffn_moe_up", il);
        GGML_ASSERT(h->ne[0] == n_ff_exp);
        ggml_tensor * experts = ggml_mul_mat_id(ctx0, layer.ffn_down_exps, h, selected);     // [n_embd, n_expert_used, n_rows]
        experts = ggml_mul(ctx0, experts, weights);
        cb(experts, "ffn_moe_weighted", il);
        ggml_tensor * routed = n_expert_used == 1
            ? ggml_cont(ctx0, ggml_view_2d(ctx0, experts, n_embd, n_rows, experts->nb[2], 0))
            : ggml_multi_add(ctx0, ggml_view_2d(ctx0, experts, n_embd, n_rows, experts->nb[2], 0), n_expert_used);

        ggml_tensor * out = ggml_add(ctx0, ggml_add(ctx0, shared, routed), inp);
        cb(out, "l_out", il);
        return out;
    };

    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    inpL = llm_build_lora_mm(lctx, ctx0, model.tab_embd_up, inpL);
    cb(inpL, "inp_up", -1);

    ggml_tensor * inp_pos  = build_inp_pos();
    ggml_tensor * mask_enc = new_mask(lctx.inp_tab_mask_enc, n_tokens, "tab_mask_enc");

    // encoder
    for (int il = 0; il < n_split; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * cur = llm_build_norm(ctx0, inpL, hparams, layer.attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * Qcur = rope(llm_build_lora_mm(lctx, ctx0, layer.wq, cur), inp_pos, hparams.n_head(il),    n_tokens);
        ggml_tensor * Kcur = rope(llm_build_lora_mm(lctx, ctx0, layer.wk, cur), inp_pos, hparams.n_head_kv(il), n_tokens);
        ggml_tensor * Vcur = llm_build_lora_mm(lctx, ctx0, layer.wv, cur);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        cur = llm_build_kv(ctx0, lctx, kv_self, gf, layer.wo, nullptr,
                Kcur, Vcur, Qcur, mask_enc, n_tokens, kv_head, n_kv, kq_scale, cb, il);
        cur = ggml_add(ctx0, cur, inpL);
        cb(cur, "attn_with_inp", il);

        inpL = moe(cur, il, n_tokens);
    }

    // one shared K/V pair per token, in the cache slot of layer n_split
    {
        ggml_tensor * cur = llm_build_norm(ctx0, inpL, hparams, model.tab_kv_norm, NULL, LLM_NORM_RMS, cb, n_split);
        cb(cur, "tab_kv_norm", -1);
        ggml_tensor * Kcur = rope(llm_build_lora_mm(lctx, ctx0, model.tab_k, cur), inp_pos, hparams.n_head_kv(n_split), n_tokens);
        ggml_tensor * Vcur = llm_build_lora_mm(lctx, ctx0, model.tab_v, cur);
        // as llm_build_kv does for a layer's own K/V
        if (cparams.k_cache_hadamard) {
            if (int block_size = model.hadamard_size_k(n_split); block_size > 0) {
                Kcur = ggml_hadamard(ctx0, Kcur, block_size);
            }
        }
        if (cparams.v_cache_hadamard) {
            if (int block_size = model.hadamard_size_v(n_split); block_size > 0) {
                Vcur = ggml_hadamard(ctx0, Vcur, block_size);
            }
        }
        cb(Kcur, "tab_K", -1);
        cb(Vcur, "tab_V", -1);
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, Vcur);
        llm_build_kv_store(lctx, ctx0, hparams, cparams, kv_self, gf, Kcur, Vcur, n_tokens, kv_head, cb, n_split);
    }

    if (n_outputs == 0) {
        // a tab, or a piece of a prompt: nothing reads the decoder state of these tokens
        if (n_tokens > 1) {
            lctx.inp_out_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);     // never allocated, never read
        }
        return gf;
    }

    // decoder, on the tokens whose logits are wanted
    ggml_tensor * x       = inpL;
    ggml_tensor * pos_out = inp_pos;
    int64_t       n_rows  = n_tokens;
    if (n_outputs < n_tokens) {
        n_rows = n_outputs;
        x = ggml_get_rows(ctx0, inpL, build_inp_out_ids());
        lctx.inp_tab_pos_out = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rows);
        cb(lctx.inp_tab_pos_out, "tab_pos_out", -1);
        ggml_set_input(lctx.inp_tab_pos_out);
        pos_out = lctx.inp_tab_pos_out;
    }
    ggml_tensor * mask_dec = new_mask(lctx.inp_tab_mask_dec, n_rows, "tab_mask_dec");

    for (int il = n_split; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * cur = llm_build_norm(ctx0, x, hparams, layer.attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * Qcur = rope(llm_build_lora_mm(lctx, ctx0, layer.wq, cur), pos_out, hparams.n_head(il), n_rows);
        cb(Qcur, "Qcur", il);

        cur = llm_build_kv(ctx0, lctx, kv_self, gf, layer.wo, nullptr,
                nullptr, nullptr, Qcur, mask_dec, n_rows, kv_head, n_kv, kq_scale, cb, il, nullptr, 0, n_split);
        cur = ggml_add(ctx0, cur, x);
        cb(cur, "attn_with_inp", il);

        x = moe(cur, il, n_rows);
    }

    // head: norm, down to the embedding width, then the embedding table
    ggml_tensor * cur = llm_build_norm(ctx0, x, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);
    cur = llm_build_lora_mm(lctx, ctx0, model.tab_embd_down, cur);
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
