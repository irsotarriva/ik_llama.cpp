# Prompt cache controlled by the client

The server decides by itself where to take context checkpoints (every 512 tokens and near the end
of a prompt) and which to drop. A client usually knows better: a chat interface knows that message
ends are where a user branches, a coding harness knows that its system prompt and tool list never
change. With the fields below a client says where the checkpoints go and which states are worth
keeping on disk, and for how long.

Nothing here changes requests that do not use the fields.

## Marking positions

On `/v1/chat/completions`, put `cache_control` on a message (or on one of its content parts):

```json
{"messages": [
  {"role": "system", "content": "...long instructions and tool descriptions...",
   "cache_control": {"type": "ephemeral", "ttl": "1h"}},
  {"role": "user", "content": "first question", "cache_control": {"type": "ephemeral"}},
  {"role": "assistant", "content": "...", "cache_control": {"type": "ephemeral"}},
  {"role": "user", "content": "second question"}
]}
```

- Each mark is a breakpoint at the end of that message. The server ends a batch exactly there and
  takes a checkpoint, and it drops its own checkpoints before the marked ones. A later request that
  differs only after that message (an edited reply, a regenerated answer, a branch) reads only what
  is new.
- A mark with `ttl` also writes the state of everything up to that message to disk, if the server
  was started with a store (below). Any later request whose prompt starts with the same tokens loads
  it instead of reading them, also after the server was restarted.
- `ttl` is seconds as a number, or a string such as `"90"`, `"5m"`, `"1h"`, `"7d"`.
  `"persist": false` keeps a mark with a `ttl` in memory only.
- A breakpoint at the very end of the prompt is not taken (in a chat request the prompt goes on
  after the last message, so the last message end is fine).
- A mark is ignored without notice when the chat template writes that message differently once
  others follow it, or when its end does not fall between two tokens.

On `/completion` and `/v1/completions` the same is given directly:

```json
{"prompt": "...", "cache_breakpoints": [{"n_tokens": 1581, "ttl": 3600}, {"chars": 20410}]}
```

`n_tokens` counts prompt tokens, `chars` counts bytes of the prompt string.

## The store

Environment variables, read at start:

| variable | meaning |
|---|---|
| `LA_CACHE_DIR=folder` | keep states that clients mark with a `ttl` in this folder. Without it marks only make checkpoints in memory |
| `LA_CACHE_MAX_GB=20` | size limit. Over it, the entries that expire soonest are deleted |

- An entry is deleted when its time is up. Every request that uses it starts its time again, with
  the lifetime it was created with.
- A state is valid for one model file and one pair of KV cache types; both are part of its name, and
  entries of other models are listed but never loaded.
- Size on Qwen3.8-Flash-Next (8-bit KV cache): 144 MB for 1,581 tokens. About 110 MB of that is the
  model's recurrent state and does not grow with the prompt; the rest is 17 KB per token.
- Lossless compression is not offered: zlib and lzma take 6% off such a file.

Requests:

| request | does |
|---|---|
| `GET /cache` | lists the entries: `id`, `n_tokens`, `bytes`, `ttl`, `expires_in`, `n_uses`, `usable` |
| `POST /cache/extend` `{"id": "...", "ttl": "1d"}` | gives an entry a new lifetime from now; without `ttl`, its old one again |
| `POST /cache/delete` `{"id": "..."}` | deletes an entry |

## What it does to time to first token

Qwen3.8-Flash-Next on 8 CPU cores, a system message of 1,581 tokens:

| | time before the first new token is computed |
|---|---|
| read the prompt | 51 to 73 s |
| state loaded from the store after a restart | 0.07 to 0.15 s to load, 2.5 s for the 22 new tokens |

The answer after the restart is the same text as before it.

## Limits

- Text prompts only. A prompt with images, a context that was shifted, and servers with a system
  prompt set at start are served as before, without the store.
- One state per position. Two different continuations of one prefix share the entry for the prefix.
- With a routing bias (`LA_BETA`) a state made under one set of resident experts is reused under
  another. That is what the model computed, but a rerun is then not bit for bit the same.
- `tests/test_cache_breakpoints.py` in the project folder runs all of this against a real model.
