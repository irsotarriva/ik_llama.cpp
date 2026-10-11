# Tabs

A tab is a piece of context that is encoded apart from the prompt: a file, a document, a tool
result. The prompt (the agent stream) reads every open tab in full, and what a tab leaves in the
cache depends on that tab's tokens only. So:

- opening a tab costs its own tokens and nothing else
- editing a tab re-encodes that tab; the other tabs and the prompt read so far stay as they are
- closing a tab costs nothing
- the order of the tabs does not matter

This needs a model trained for it. The engine runs the split tab architecture (`tabsplit`,
`src/graphs/build_tabsplit.cpp`): the first layers encode each segment alone, then every token gets
one K/V pair, and the remaining layers only read those. On any other model a request with `tabs`
is refused.

## Starting the server

| variable | meaning |
|---|---|
| `LA_TABS=N` | the most tabs kept at a time, over all slots. Without it tabs are off |
| `LA_TAB_CTX=M` | cells of the context (`-c`) set aside for tabs. Default: half. The slots share the rest |
| `LA_CACHE_DIR=folder` | also keep tabs that ask for it on disk (the store of `README-cache.md`) |

## A request

`tabs` on `/completion`, `/v1/completions` and `/v1/chat/completions`:

```json
{"prompt": "Qn143A",
 "tabs": [
   {"name": "config", "content": "<tab>n124=n41\nn185=c20\n..."},
   {"name": "users",  "content": [1, 131, 2, 48, 3], "ttl": "1h"}
 ]}
```

- The list is the set of tabs that are open for this request. A request without `tabs` has none.
- `content` is text or token ids. No BOS is added: a tab says itself how it starts.
- A tab is recognised by its tokens. `name` is only a label that is reported back.
- A tab that is already in memory is attached without any computation, whichever slot or request
  put it there. A tab the slot had and the request does not list is detached; it stays in memory
  until its cells are needed for another one (the least recently used goes first).
- `ttl` (as in `README-cache.md`: seconds, or `"5m"`, `"1h"`, `"7d"`) also writes the tab to the
  store on disk. After a restart it is loaded instead of encoded. Every use starts its time again.
- What the slot holds of the prompt is reused as always: it does not depend on the tabs. Only the
  tokens that are still to be computed read them, and at least the last prompt token always is.
- More tabs than `LA_TABS`, or more tokens than `LA_TAB_CTX`, in one request: error 400.

The response says what was done, in `timings`:

```json
"tabs": {"open": 6, "reused": 5, "loaded": 0, "encoded": 1, "tokens_encoded": 25, "ms": 1.2}
```

`GET /tabs` lists the tabs in memory: `id`, `name`, `n_tokens`, `seq`, the `slots` that have it
open, `last_used`, `n_uses`, `on_disk`, and the cells set aside and in use. Tabs on disk are in
`GET /cache` with `"kind": "tab"`, and `/cache/extend` and `/cache/delete` work on them.

## In the library

```c
llama_tab_set_agent(ctx, agent_seq, true);        // once, before its first token
// a tab: a sequence of its own, positions from 0, no logits
llama_decode(ctx, tab_batch);
llama_tab_attach(ctx, agent_seq, tab_seq);
// ... decode the agent's tokens with positions from 0; the model's offset is added inside
llama_tab_detach(ctx, agent_seq, tab_seq);        // close
llama_kv_cache_seq_rm(ctx, tab_seq, -1, -1);      // and free its cells
```

An edit is `llama_kv_cache_seq_rm` on the tab's sequence and a decode of the new tokens.
`examples/tab-test/tab-ref.cpp` does all of it against the logits of the training code.

## Limits

- The 16-bit K/V cache moves this model's logits by up to 1 on a range of 130 (0.005 with
  `-ctk f32 -ctv f32 -no-fa`). The answers of the test model are the same either way.
- A tab keeps the K/V of the encoder layers as well as the shared pair, in memory and on disk,
  although nothing reads them once the tab is encoded: 4 KB per token for the test model where 1 KB
  would do.
- The decoder runs only on tokens whose logits are asked for, so a prompt costs the encoder layers
  but for its last token. Embeddings of the last layer for every token are therefore not available.
- No context shift for tabs, and the slot's own context shift does not know about them.
- The test model has no chat template; the chat endpoint passes `tabs` on but was not run.
