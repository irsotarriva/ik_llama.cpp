# Load-aware expert cache: settings

This fork of ik_llama.cpp (branch `load-aware`) can keep only part of a mixture-of-experts model's
routed experts in memory and read the others from the GGUF when the router asks for them. Everything
is switched with environment variables, read once at start by `llama-cli`, `llama-server` and
`llama-perplexity`. Without `LA_ENABLE=1` the engine behaves as upstream.

Measurements behind the advice here are in the project log, `I:\localAIOptimization\RESULTS.md`.

## The settings that matter

| variable | meaning |
|---|---|
| `LA_ENABLE=1` | turn the load-aware selection on |
| `LA_REAL=1` | real cache: resident experts in the engine's own memory, the rest read from the file with unbuffered I/O. Without it residency is only modelled (for quality measurements; needs the whole model in RAM or page cache) |
| `LA_RESIDENT_FRAC=0.58` | share of the routed experts kept in RAM (0 to 1) |
| `LA_BETA=0.25` | bias toward resident experts, in units of the router's top-k probability range. 0 = exactly what the router asks for. Most of the speed is won by 0.25; quality collapses between 1.0 and 1.5 on both models tried |
| `LA_PROTECT=2` | always serve the router's first N experts, whatever the bias. With `LA_PROTECT=2 LA_BETA=4` every other miss is substituted and there is no cliff |
| `LA_PROFILE=file` | usage profile: which experts to load before the first token. Without one the cache starts empty |
| `LA_PROFILE_OUT=file` | write the profile of this run (one float per layer and expert). Rewritten every 512 tokens and at exit. If `LA_PROFILE` is also given, the old profile counts as 4,096 tokens of earlier use, so the same path for both keeps a running profile |
| `LA_GPU_FRAC=0.5` | share of the experts copied into VRAM slots and run on the GPU (needs a profile and a GPU build; dense layers with `-ngl 99 -ot exps=CPU`) |

Others, with their defaults: `LA_THETA` (0; drop weak experts, not useful on stock models),
`LA_GAMMA` (0; multiplicative form of the bias), `LA_EVICT` (`freq`; `lru` for least recently used),
`LA_DECAY` (decay of the request counts), `LA_PRICE_RESIDENT` (0.1), `LA_EXCESS_RAM` (0; extra price
for RAM against VRAM), `LA_IO_THREADS` (8), `LA_PREFETCH` (0; read one layer ahead), `LA_ASYNC`,
`LA_ASYNC_RATE`, `LA_ASYNC_QUEUE`, `LA_ASYNC_THREADS` (background loading of substituted experts,
costs speed in real mode), `LA_STATS=file` (append one JSON line of counters at exit).

## Command-line options to use with it

- `--defer-ple` for models with a large embedding lookup table (Qwen3.8-Flash-Next: 27 GiB). Without
  it the engine reads every page of the table into memory at load on Windows.
- `--no-warmup` whenever the model does not fit in RAM and `LA_REAL` is off: the warm-up runs every
  expert once, which reads the whole file. In real mode the warm-up is harmless.
- `llama-server`: `--cache-ram` (default 8192 MiB) and `--ctx-checkpoints` (default 32, about 60 to
  120 MiB each) are memory outside the cache's budget; count them when sizing `LA_RESIDENT_FRAC`.
- Do not use `--mlock` with `LA_REAL`: the cache owns the expert memory.
- GPU builds: `GGML_CUDA_NO_PINNED_WEIGHTS=1` with `-ot exps=CPU`, or every CPU-side weight is copied
  into pinned memory.

## Sizing the resident share

Private memory of the process is, to within a few hundred MiB:
`LA_RESIDENT_FRAC` x routed experts + KV cache + compute buffers (about 1.4 GiB) + checkpoints and
prompt cache as they fill. The dense weights (a few GiB) are mapped from the file on top of that.
Leave what the other programs on the machine use, measured, not guessed: a server sized past it
pushes them into the page file.

## Making a profile

    LA_ENABLE=1 LA_PROFILE_OUT=profile.f32 llama-perplexity -m model.gguf -f some_text.txt -c 512 --chunks 8 --no-warmup

or run the server with `LA_PROFILE_OUT` and use it. The file is `n_layers x n_expert` float32 values.

## What the end-of-run lines mean

    moe-la: events=18576 served/event=10.000 reads/event=0.710 exact_reads/event=0.710 substituted/event=0.000 ...
    moe-la: read 29.88 GiB in 32.07 s of waiting (0.93 GiB/s), 1513 evictions; prefill 0.00 GiB in 0.00 s

An event is one layer of one token. `reads/event` times the number of layers is disk reads per token;
`exact_reads` is what exact routing would have read; `substituted` is how many of the router's choices
were replaced by resident experts. `prefill` is what the profile loaded before the first token.
