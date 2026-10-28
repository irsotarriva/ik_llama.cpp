# Load-aware recipes

Measured selector settings per model family, so a result can be reproduced without remembering
which variables it took. The engine uses the recipes in this folder of the source tree it was built
from; `LA_RECIPES=path/to/recipes` points it elsewhere.

At load the engine reads every `*.recipe` here, takes the one that matches the model best (the most
`match` keys that hold; `default.recipe` matches anything) and sets its variables. It prints what
it chose:

    la-recipe: qwen35moe.recipe, level 'exact': LA_PROFILE=.../qwen3.6-35b-a3b.f32 LA_BETA=0

- `--quality exact|near|balanced|fast` (or `LA_RECIPE_LEVEL`) picks a level; without it `exact`.
- A variable already set by hand is kept (`la-recipe: given by hand, kept: ...`).
- `LA_RECIPES` may also name one file.
- A recipe does not decide whether the cache runs or how much memory it gets: that depends on the
  machine and is chosen at load (`--experts-ram`, `--experts-vram`; `LOAD_AWARE.md`).

## Levels

A level is a quality budget: KL above exact routing allowed on every model of the family measured,
at 15% and at 40% of the experts resident. At each level the setting is the one that saved the most
disk reads in its worst case, and a setting that swaps experts has to save at least 10% to be
chosen over exact routing.

| level | KL budget | Qwen3.6 family | Qwen3.8-Flash-Next | Gemma4-26B-A4B | LFM2-24B-A2B | default |
|---|---|---|---|---|---|---|
| exact | 0 | exact | exact | exact | exact | exact |
| near | 0.005 | exact | exact | exact | exact | exact |
| balanced | 0.02 | `LA_WCAP=0.2` (54% saved worst case, 79% at 40% resident) | `LA_WCAP=0.1` (38%, 63%) | exact | exact | exact |
| fast | 0.1 | `LA_BETA=1` (87%, 96%) | `LA_BETA=1` (93%, 97%) | `LA_WCAP=0.1` (30%, 46%) | exact | exact |

Defaults: `balanced` for the two Qwen families, `exact` for the others. LFM2 loses quality with any
swap, so its speed has to come from memory and the VRAM tier. No single setting held on all five
models, so `default.recipe` routes exactly at every level. "near" is exact everywhere only because
the grid has no point between exact and a cap of 0.05 or a bias of 0.25.

The worst case is usually the 15% row. With more of the model in memory a stronger setting fits the
same budget: at 40% resident, Qwen3.8 with `LA_BETA=0.25` costs +0.014 KL for 82% of reads saved and
Qwen3.6 +0.007 for 75%. Give it by hand when the resident share is that high.

Qwen3.8 was measured in real mode (`LA_REAL=1`, the file is larger than RAM) with batches of 16 and
no read-ahead, the way it generates; its base is exact routing on that same path.

## Making and changing them

The recipes come from measured grids of quality (KL divergence against exact routing) and disk
reads saved, per setting and model. The scripts that produce them are not part of this repository
yet. To add a model by hand: copy a recipe, change its `match` keys, and record a usage profile
with `LA_PROFILE_OUT`.

The profile in a family recipe comes from its reference model; a finetune routes much the same, but
its own profile (`LA_PROFILE_OUT`) is better and, given by hand, wins.
