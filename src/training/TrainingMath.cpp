#include "training/TrainingMath.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace training_math {

std::vector<float> normalized_ranks(const std::vector<float>& f) {
    size_t n = f.size();
    if (n == 0) return {};

    // Sort indices descending by fitness; tie-break ascending by index.
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (f[a] != f[b]) return f[a] > f[b]; // higher fitness → lower rank index (better)
        return a < b;                           // lower car index wins on tie
    });
    // order[rank] = car index; rank 0 = best
    // r = (n-1-rank) / (n-1)  →  rank 0 → r=1, rank n-1 → r=0

    std::vector<float> result(n);
    float denom = (n <= 1) ? 1.f : (float)(n - 1);
    for (size_t rank = 0; rank < n; ++rank)
        result[order[rank]] = (float)(n - 1 - rank) / denom;
    return result;
}

float cvar(const std::vector<float>& x, float alpha) {
    size_t n = x.size();
    if (n == 0) return 0.f;
    size_t k = (size_t)std::ceil(alpha * (float)n);
    if (k < 1) k = 1;
    if (k > n) k = n;

    std::vector<float> tmp = x;
    std::nth_element(tmp.begin(), tmp.begin() + (long)k, tmp.end());
    // Elements [0, k) are the k smallest (unordered)
    float sum = 0.f;
    for (size_t i = 0; i < k; ++i) sum += tmp[i];
    return sum / (float)k;
}

std::vector<float> normalize_zscore(const std::vector<float>& f) {
    size_t n = f.size();
    if (n == 0) return {};
    float mu = 0.f;
    for (float v : f) mu += v;
    mu /= (float)n;
    float var = 0.f;
    for (float v : f) { float d = v - mu; var += d * d; }
    float sigma = std::sqrt(var / (float)n);
    constexpr float eps = 1e-6f;
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = (f[i] - mu) / std::max(sigma, eps);
    return out;
}

std::vector<float> normalize_minmax(const std::vector<float>& f) {
    size_t n = f.size();
    if (n == 0) return {};
    float mn = *std::min_element(f.begin(), f.end());
    float mx = *std::max_element(f.begin(), f.end());
    constexpr float eps = 1e-6f;
    float range = std::max(mx - mn, eps);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = (f[i] - mn) / range;
    return out;
}

std::vector<float> normalize_progress(const std::vector<float>& f, float wProgress) {
    float norm = std::max(1.f, wProgress);
    std::vector<float> out(f.size());
    for (size_t i = 0; i < f.size(); ++i)
        out[i] = f[i] / norm;
    return out;
}

std::vector<int> active_map_indices(int gen, int total, const CurriculumConfig& cfg) {
    if (total <= 0) return {};

    int count = 0;
    switch (cfg.mode) {
        case CurriculumMode::None:
            count = total;
            break;
        case CurriculumMode::Linear:
            count = std::min(std::max(cfg.start + gen / cfg.step, 1), total);
            break;
        case CurriculumMode::Explicit: {
            int n = 0;
            for (int thresh : cfg.schedule)
                if (gen >= thresh) ++n;
            count = std::min(std::max(1 + n, 1), total);
            break;
        }
    }

    // Start with sequential indices [0, count)
    std::vector<int> idx;
    idx.reserve((size_t)total);
    for (int i = 0; i < count; ++i) idx.push_back(i);

    // Merge pinned indices (clamped to [0, total))
    for (int p : cfg.pinned) {
        if (p >= 0 && p < total) {
            bool found = false;
            for (int x : idx) if (x == p) { found = true; break; }
            if (!found) idx.push_back(p);
        }
    }

    std::sort(idx.begin(), idx.end());
    return idx;
}

} // namespace training_math
