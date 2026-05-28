#pragma once
#include "Training.h"
#include <vector>

namespace training_math {

// R1: normalized ranks for one map's fitnesses.
// r[c] ∈ [0, 1] where 1 = best. Tie-break: lower car index wins.
std::vector<float> normalized_ranks(const std::vector<float>& f);

// CVaR_α: mean of the ceil(α * n) smallest values of x. α ∈ (0, 1].
float cvar(const std::vector<float>& x, float alpha);

// R3: per-map normalizations
std::vector<float> normalize_zscore(const std::vector<float>& f);
std::vector<float> normalize_minmax(const std::vector<float>& f);
std::vector<float> normalize_progress(const std::vector<float>& f, float wProgress);

// R2: number of active maps for generation gen given the curriculum config.
int active_map_count(int gen, int total, const CurriculumConfig& cfg);

} // namespace training_math
