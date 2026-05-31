#include "core/TrackGen.h"
#include "core/Constants.h"
#include <algorithm>
#include <sstream>
#include <random>
#include <cmath>

namespace trackgen {

// --- Augmentation transforms (Etapa 4) -------------------------------------
TrackData mirrorX(const TrackData& base) {
    TrackData d = base;
    d.name = base.name + "_mirror";
    if (base.waypoints.empty()) return d;
    float minX = base.waypoints[0].x, maxX = minX;
    for (const auto& w : base.waypoints) { minX = std::min(minX, w.x); maxX = std::max(maxX, w.x); }
    float cx = 0.5f * (minX + maxX);
    for (auto& w : d.waypoints)  w.x   = 2.f * cx - w.x;
    for (auto& o : d.obstacles)  o.pos.x = 2.f * cx - o.pos.x;
    return d;
}

TrackData reverse(const TrackData& base) {
    TrackData d = base;
    d.name = base.name + "_rev";
    std::reverse(d.waypoints.begin(), d.waypoints.end());
    return d;
}

TrackData scaleWidth(const TrackData& base, float factor) {
    TrackData d = base;
    d.trackWidth = base.trackWidth * factor;
    std::ostringstream ss;
    ss << base.name << "_w" << factor;
    d.name = ss.str();
    return d;
}

// --- Procedural generation (Etapa 5) ---------------------------------------

// True if the track ribbon crosses itself: two centerline points far apart along the
// path but within one track-width of each other (their half-width ribbons overlap).
static bool selfOverlaps(const Track& t, float width) {
    const auto& c   = t.centerline();
    const auto& arc = t.arcLength();
    const float L   = t.totalArcLength();
    const float gate2     = width * width;       // centerlines within `width` → ribbons touch
    const float minArcSep = width * 2.0f;        // ignore neighbours close along the path
    const size_t n = c.size();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            float along = arc[j] - arc[i];
            float sep   = std::min(along, L - along); // closed-loop arc distance
            if (sep < minArcSep) continue;
            float dx = c[i].x - c[j].x, dy = c[i].y - c[j].y;
            if (dx * dx + dy * dy < gate2) return true;
        }
    }
    return false;
}

// True if the full car footprint fits inside the ribbon at every point along the
// centerline (heading = local tangent). This is exactly the collision test the
// simulation runs (center + 4 corners), so a loop passing this never collides at
// spawn and stays drivable through every corner. Robust to float seam issues since
// the corners are off the centerline.
static bool footprintInside(const Track& t) {
    const auto& c   = t.centerline();
    const auto& arc = t.arcLength();
    const float hl = CAR_LENGTH * 0.5f, hw = CAR_WIDTH * 0.5f;
    for (size_t i = 0; i < c.size(); ++i) {
        Vec2 tan = t.tangentAtArc(arc[i]);
        float ca = tan.x, sa = tan.y; // unit tangent = heading at this point
        Vec2 pts[4] = {
            {c[i].x + ca*hl - sa*hw, c[i].y + sa*hl + ca*hw},
            {c[i].x + ca*hl + sa*hw, c[i].y + sa*hl - ca*hw},
            {c[i].x - ca*hl - sa*hw, c[i].y - sa*hl + ca*hw},
            {c[i].x - ca*hl + sa*hw, c[i].y - sa*hl - ca*hw},
        };
        for (const auto& q : pts) if (!t.isInsideTrack(q)) return false;
    }
    return true;
}

static TrackData sampleLoop(std::mt19937& rng, const GenParams& p) {
    std::uniform_real_distribution<float> U01(0.f, 1.f);
    int span = std::max(1, p.maxWaypoints - p.minWaypoints + 1);
    int N = p.minWaypoints + (int)(rng() % (unsigned)span);
    if (N < 3) N = 3;

    TrackData d;
    d.closed     = true;
    d.trackWidth = p.widthMin + U01(rng) * (p.widthMax - p.widthMin);

    // Radius as a sum of low-frequency harmonics of the angle: r(θ) is single-valued,
    // so the loop is star-convex (never self-intersects) and smooth (gentle corners).
    // Amplitudes scale with radialJitter and shrink with frequency to bound curvature.
    // Low frequencies only (2,3): higher harmonics alias at N≈10-18 samples and create
    // sharp Catmull-Rom wiggles. Amplitude decays as 1/(k+1)² to keep curvature gentle.
    const int   K = 2;
    const int   freq[K] = {2, 3};
    float amp[K], phase[K];
    for (int k = 0; k < K; ++k) {
        amp[k]   = p.radialJitter * U01(rng) / (float)((k + 1) * (k + 1));
        phase[k] = 2.f * (float)M_PI * U01(rng);
    }
    for (int i = 0; i < N; ++i) {
        float theta = 2.f * (float)M_PI * (float)i / (float)N;
        float mod = 0.f;
        for (int k = 0; k < K; ++k)
            mod += amp[k] * std::sin((float)freq[k] * theta + phase[k]);
        float r = p.baseRadius * (1.f + mod);
        d.waypoints.push_back({ p.center.x + std::cos(theta) * r,
                                p.center.y + std::sin(theta) * r });
    }
    return d;
}

static TrackData ellipseFallback(const GenParams& p) {
    TrackData d;
    d.closed     = true;
    d.trackWidth = 0.5f * (p.widthMin + p.widthMax);
    const int N = 12;
    for (int i = 0; i < N; ++i) {
        float ang = 2.f * (float)M_PI * (float)i / (float)N;
        d.waypoints.push_back({ p.center.x + std::cos(ang) * p.baseRadius,
                                p.center.y + std::sin(ang) * p.baseRadius * 0.7f });
    }
    d.name = "proc_fallback";
    return d;
}

TrackData generateLoop(std::uint32_t seed, const GenParams& params) {
    std::mt19937 rng(seed);
    int attempts = std::max(1, params.maxAttempts);
    for (int a = 0; a < attempts; ++a) {
        TrackData d = sampleLoop(rng, params);
        std::ostringstream ss; ss << "proc_" << seed;
        d.name = ss.str();
        try {
            Track t(d);
            // footprintInside covers the 4 car corners along the whole path; the car
            // also tests its center, and at spawn the center sits exactly on the
            // centerline (float-fragile seam) — require it explicitly so the car never
            // collides on tick 0.
            if (t.totalArcLength() > 0.f && !selfOverlaps(t, d.trackWidth)
                && footprintInside(t) && t.isInsideTrack(t.spawnPos()))
                return d;
        } catch (...) {
            // degenerate sample (e.g. coincident points) — try again
        }
    }
    // Deterministic safe fallback if no clean loop was found within the budget.
    TrackData fb = ellipseFallback(params);
    std::ostringstream ss; ss << "proc_" << seed << "_fb";
    fb.name = ss.str();
    return fb;
}

} // namespace trackgen
