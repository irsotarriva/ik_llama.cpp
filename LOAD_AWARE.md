# Load-aware expert cache

This fork of ik_llama.cpp can keep only part of a mixture-of-experts model's routed experts in memory
and read the others from the GGUF when the router asks for them, so a model larger than RAM runs at a
usable speed instead of thrashing the disk.

## Using it

It is on by default and steps in only when needed. Once the weights are placed, the engine adds up the
routed experts that run on the CPU:

- if they fit in the RAM set aside for them, nothing changes: the engine runs exactly as ik_llama.cpp;
- if they do not, the most used experts are kept in that RAM and the others are read from disk when
  the router asks for them. At the default quality, `exact`, every token still gets the experts the
  router chose: the output is the same as with the whole model in memory, only slower.

| flag | meaning |
|---|---|
| `--experts-ram SIZE` | RAM for the routed experts, such as `24G` or `512M`. Default `auto`: the memory free at load, less the other weights and a reserve of the larger of 10% and 2 GiB, which is kept free while the model runs |
| `--experts-vram SIZE` | VRAM for the most used experts, such as `4G`, or one value per GPU in device order (`6G,0`). Default `auto`: what the main GPU has left once its weights, KV cache and compute buffers are counted, less `--experts-vram-margin`. Used only when the cache is on, and only with a usage profile (a recipe gives one). Put the dense layers on the GPU and the experts on the CPU: `-ngl 99 --cpu-moe` |
| `--experts-vram-margin SIZE` | VRAM the auto budget leaves free on each GPU, one value or one per GPU. Default 1 GiB on a GPU that drives a display (the Windows driver quietly moves memory to system RAM when the card is full, which slows everything down) and 512 MiB on others |
| `--quality LEVEL` | `exact` (default): always the router's experts. `near`, `balanced`, `fast`: a token may be served an expert already in memory instead of one that would be read from disk, for less quality and fewer reads. Only matters when experts are read from disk |
| `--expert-cache MODE` | `auto` (default), `on` (also when everything fits, for recording a usage profile) or `off` (never; the engine as ik_llama.cpp) |

Examples:

```
llama-server -m model.gguf                                   # auto: on only if the experts do not fit
llama-server -m model.gguf --experts-ram 20G --quality near  # 20 GiB for experts, may swap a few
llama-server -m model.gguf -ngl 99 --cpu-moe --experts-vram 6G
llama-server -m model.gguf --expert-cache off                # as ik_llama.cpp
llama-server -m model.gguf --config engine.toml              # the same settings from a file
```

Reading from disk needs the model memory-mapped (no `--no-mmap`). The log says what was chosen:
`expert cache: on, 13.2 GiB of experts: 6.8 GiB in RAM (auto), 0.0 GiB in VRAM, the rest read from disk; quality exact`.

The flags are read by the programs that use the common arguments (`llama-server`, `llama-cli`,
`llama-perplexity`, ...). For others, such as `llama-bench`, the same settings are environment variables:
`LA_EXPERT_CACHE`, `LA_EXPERTS_RAM`, `LA_EXPERTS_VRAM`, `LA_VRAM_MARGIN` and `LA_RECIPE_LEVEL`.

A VRAM budget on a GPU other than the main one (`--main-gpu`) is refused for now: the expert slots
live on the main GPU. Placing each layer's slots on the GPU that holds that layer is planned.

With `--fit`, a VRAM budget given as a size is kept out of the fit, so the fit leaves that much free
for the slots; the automatic budget takes what the fit leaves.

Measured settings per model family are kept as recipes (`recipes/README.md`), found in the `recipes`
folder of the source tree or at `LA_RECIPES=path/to/recipes`. They give the usage profile (which
experts to load first) and what each quality level does on that family.

## Config file

The same settings can be kept in a TOML file, given with `--config engine.toml`. Every key is
optional: one that is missing is chosen at load, as without the file. Flags given as well win over
the file, and so do `LA_*` variables already set in the environment.

```toml
version = 1

[experts]
cache   = "auto"        # --expert-cache: auto, on or off
quality = "exact"       # --quality: exact, near, balanced or fast
ram     = "auto"        # --experts-ram: auto or a size ("24G"; a number is GiB)

[devices.CUDA0]         # one section per GPU, named as the backend names it (CUDA0, CUDA1, ...)
vram        = "auto"    # --experts-vram
vram_margin = "1G"      # --experts-vram-margin

[cache_disk]            # the server's prompt cache on disk
dir = "prompt-cache"    # --cache-disk (relative to the file's folder)
max = "20G"             # --cache-disk-max
```

`--print-config` reads the model's metadata (no weights), works out every setting the way a real
start would, prints it as such a file and exits. Each automatic choice is filled in, with a comment
on how it was made, so the output can be saved, edited and given back with `--config`:

```
llama-server -m LFM2-24B-A2B-Q4_K_M.gguf -ngl 99 --cpu-moe --experts-ram 6G --print-config > engine.toml
```

```toml
[experts]
# auto: the 12.9 GiB of experts do not fit the 6.0 GiB of RAM for them, the rest is read from disk
cache = "on"
quality = "exact"
ram = "6G"

[devices.CUDA0]   # NVIDIA GeForce RTX 3060
# auto: 11.0 GiB free, less 0.5 GiB of weights, 2.6 GiB of KV cache and compute buffers and a 1G margin (it drives a display)
vram = "6.89G"
vram_margin = "1G"
```

The KV cache counted is the one the context will allocate: with no `-c`, the model's full training
context. The slots are sized by walking the usage profile's ranking with each expert's real size, so
they stay within the budget when layers use different quantization types.

A saved file pins the choices: the automatic ones are worked out again only for keys left out.
Files ending in `.json` are read as JSON with the same layout, which is also the form a program can
use to pass the config around.

Unknown keys are errors, so a typo does not go unnoticed. `layers` tables (under `[experts]`, a device
or at the top) are kept for placement per layer in a later version and are refused for now.

## Tuning by hand

Everything below is for experiments and measurements. Setting `LA_ENABLE` (to 1 or 0) switches the
automatic choice off: the variables are then used as given, as fractions of the routed experts.


| variable | meaning |
|---|---|
| `LA_ENABLE=1` | turn the load-aware selection on (0: off), with no automatic choice |
| `LA_REAL=1` | real cache: resident experts in the engine's own memory, the rest read from the file with unbuffered I/O. Without it residency is only modelled (for quality measurements; needs the whole model in RAM or page cache) |
| `LA_RESIDENT_FRAC=0.58` | share of the routed experts kept in RAM (0 to 1) |
| `LA_BETA=0.25` | bias toward resident experts, in units of the router's top-k probability range. 0 = exactly what the router asks for. Most of the speed is won by 0.25; quality collapses between 1.0 and 1.5 on both models tried |
| `LA_PROTECT=2` | always serve the router's first N experts, whatever the bias. With `LA_PROTECT=2 LA_BETA=4` every other miss is substituted and there is no cliff |
| `LA_WCAP=0.2` | instead of `LA_BETA`: a token swaps experts that are not in memory for ones that are while its mix of experts moves by at most this share (total-variation distance of the normalized weights). Same frontier as the bias on Qwen3.6, better on Gemma4 up to half the reads saved; the quality cost of a value still differs between models |
| `LA_PROFILE=file` | usage profile: which experts to load before the first token. Without one the cache starts empty |
| `LA_PROFILE_OUT=file` | write the profile of this run (one float per layer and expert). Rewritten every 512 tokens and at exit. If `LA_PROFILE` is also given, the old profile counts as 4,096 tokens of earlier use, so the same path for both keeps a running profile |
| `LA_GPU_FRAC=0.5` | share of the experts copied into VRAM slots and run on the GPU (needs a profile and a GPU build; dense layers with `-ngl 99 -ot exps=CPU`) |

Reading a prompt (real mode, on by default):

| variable | meaning |
|---|---|
| `LA_AHEAD=-1` | while a layer computes a batch, the experts the next layer will probably ask for are read: its router is applied to this layer's input and the union of the first N guesses per token is fetched. -1 = as many as a token uses, 0 = off. More than that reads too much that is not used |
| `LA_READ_MB=8` | neighbouring experts are read with one request of at most this size |

VRAM tier:

| variable | meaning |
|---|---|
| `LA_GPU_BATCH=2048` | prompt batches of this many tokens or more (0 = never). Without `LA_REAL` they skip the tier and take the engine's own GPU route for big batches, which copies the layer's experts to the card per batch: about twice as fast with a prompt in one batch, slower in batches of 1,024, so use `-b 4096 -ub 4096` or more on a GPU machine. With `LA_REAL` that route is closed (the experts are only partly in memory); giving the variable then sends the experts a batch needs through scratch tensors on the GPU, which was 19% slower than the CPU side in batches of 1,024 and 3 to 18% faster with one batch |
| `LA_GPU_ADAPT=8` | off by default. Every N generated tokens a layer may give the slot of its least requested slot expert to its most requested expert without one (`LA_GPU_ADAPT_MARGIN`, 1.5: how many times as often). In real mode the two change places in memory, nothing is read from disk |
| `LA_FAST=1` | on by default (CUDA builds). Generation and batches of up to `LA_FAST_MAX` (8) tokens take the fast path when nothing can move an expert (no `LA_REAL`, `LA_GPU_ADAPT`, `LA_WCAP`, `LA_THETA`, `LA_ASYNC`, `LA_PREFETCH`, and every expert in a slot or in RAM): the selection and the slot ids run on the GPU, and a worker thread computes the experts without a slot while the GPU runs the slots, one round trip per layer. On Qwen3.6 with the RTX 3060: 49 tok/s against 33 for the old tier path, the same text |
| `LA_FAST_THREADS` | threads of the fast path's worker (default 8 on a 16-thread CPU, half the threads otherwise); 4 to 8 are the same on the 5800X. The engine's own generation threads then have nothing to do: use `-t 1 -tb 8` (8 for prompts). Unless `OMP_WAIT_POLICY` is set, the fast path sets it to `passive`, or the idle threads spin and starve the worker (the server generated at 10 tok/s) |
| `LA_FAST_REPORT=N` | print the worker's averages every N jobs (a server is not stopped cleanly, so the summary at exit is not seen) |
| `LA_FAST_SKIP=1` | a layer whose experts all run in slots sends no job to the worker |
| `LA_EXCESS_RAM` | with the VRAM tier, how much an expert in RAM counts as missing for the bias (0 to 1). `LA_EXCESS_RAM=1` with a bias steers the router toward the experts in VRAM: with everything in RAM it is the lever for speed. Qwen3.6, 50% in VRAM: `LA_EXCESS_RAM=1 LA_PROTECT=2 LA_BETA=4` 60 tok/s at +1.1% perplexity |

Others, with their defaults: `LA_THETA` (0; drop weak experts, not useful on stock models),
`LA_GAMMA` (0; multiplicative form of the bias), `LA_EVICT` (`freq`; `lru` for least recently used),
`LA_DECAY` (decay of the request counts), `LA_PRICE_RESIDENT` (0.1), `LA_EXCESS_RAM` (0; extra price
for RAM against VRAM), `LA_IO_THREADS` (8), `LA_PREFETCH` (0; read one layer ahead), `LA_ASYNC`,
`LA_ASYNC_RATE`, `LA_ASYNC_QUEUE`, `LA_ASYNC_THREADS` (background loading of substituted experts,
costs speed in real mode), `LA_STATS=file` (append one JSON line of counters at exit),
`LA_GUARD` (0; Linux, real mode: 1 makes an expert that is not in memory fault when read, at the
price of one memory mapping per expert range, which a model with more than about 60,000 ranges
cannot afford).

Memory (real mode; added 2026-10-14 after Magnitude's memory design, not yet measured):

| variable | meaning |
|---|---|
| `LA_HEADROOM_MB=2048` | off by default. Keep this much memory available to the system. A watcher reads the available memory every `LA_HEADROOM_MS` (100); below the reserve it lowers the experts' budget, one expert per layer in turn, and gives the coldest back at once; with half the reserve (at least 512 MiB) to spare again it raises it, at most 256 MiB a check and never past the budget of the start. The experts come back as they are asked for. A layer keeps at least `LA_HEADROOM_FLOOR` (0.05) of its starting budget. The layer whose kernel may be running is trimmed at its next turn. Available memory: Windows `ullAvailPhys`, Linux `MemAvailable`, macOS free + inactive + purgeable pages |
| `LA_WIRE=1` | off by default; not on Windows. Lock the pages of resident experts (`mlock`) so the system cannot compress or swap them. Written for macOS, where the CPU-only real cache generated at 0.04 tok/s (0.31 stock) on the 8 GB M1 without explanation. A lock that fails is counted and reported (Linux allows 8 MiB by default, `ulimit -l`) |

The end-of-run lines now also give the time spent in page commits and releases (one system call per
expert read or evicted, all threads together), and with `LA_HEADROOM_MB` the least memory available
seen and how many experts the budgets were lowered and raised by.

## The lookup table (`--defer-ple`)

A token needs 16 rows of the table of Qwen3.8-Flash-Next, each a page of the file. Pages used before
stay in spare memory as long as the system has any; these settings cover the rest:

| variable | meaning |
|---|---|
| `LA_ROW_THREADS=8` | the rows of a batch are asked for at once on this many threads before the layers run. 0 = off |
| `LA_ROW_PROFILE=file` | one count per page of the table: how often it was needed. Read at start, rewritten every 1,024 tokens. 7 MB for the 27 GiB table |
| `LA_ROW_RAM_MB=2048` | at start, read the most needed pages of that profile in the background, up to this much. Default 0. The pages stay file-backed: the system drops them for free when memory is short, so they need not be counted in the budget |

They do not depend on `LA_ENABLE`.

## Command-line options to use with it

- `llama-server -ub 1024` (with `-b 2048`): prompt chunks of 1,024 tokens instead of 512. The larger the
  chunk the fewer times the experts are read and the better the kernels do: on Qwen3.8-Flash-Next with
  8 cores, 26.8 tok/s at 512, 34.7 at 1,024, 36.4 at 2,048, 41.3 at 4,096, for 1.35 GiB of compute
  buffer per 512 tokens of chunk.

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
`LA_RESIDENT_FRAC` x routed experts + KV cache + compute buffers (about 1.4 GiB per 512 tokens of
`-ub`) + checkpoints and prompt cache as they fill. During a prompt batch the cache also holds what
two layers need beyond their share (about 1 GiB on Qwen3.8-Flash-Next at 55% resident). The dense weights (a few GiB) are mapped from the file on top of that.
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

    moe-la: batch of 1024 tokens: 27.5 s, of which 2.3 s waiting for the disk; read 3.23 GiB after routing (30.0 experts/layer) and 10.13 GiB ahead (1308 experts not used)

One line per prompt batch of 32 tokens or more. `after routing` is what the guess for the next layer
missed and the batch had to wait for; `not used` is what the guess read for nothing.
