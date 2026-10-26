#pragma once

#include <string>

struct llama_model;

// Load-aware recipes: measured settings per model family, picked from the model's metadata.
// LA_RECIPES=dir (or one file) turns them on; LA_RECIPE_LEVEL picks a level (else the file's default).
// A recipe only sets LA_* variables that are not already set, so anything given by hand wins.
// Called once, after the metadata is read and before any tensor is loaded (the selector reads its
// settings when the first expert tensor is registered). See recipes/README.md.
void llama_la_apply_recipe(const llama_model & model, const std::string & fname);
