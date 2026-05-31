#pragma once
#include "core/Vec2.h"
#include "Track.h"   // TrackData
#include <cstdint>

// Procedural track generation and cheap data augmentation.
// SFML-free (core/): operates purely on TrackData, which Track can build in memory.
namespace trackgen {

struct GenParams {
    int   minWaypoints = 10;
    int   maxWaypoints = 18;
    float baseRadius   = 300.f;   // mean loop radius (px)
    float radialJitter = 0.22f;   // fractional radius variation per waypoint
    float widthMin     = 55.f;    // sampled track width range (px)
    float widthMax     = 110.f;
    Vec2  center       = {450.f, 400.f};
    int   maxAttempts  = 60;      // re-sample budget until a valid loop is found
};

// --- Augmentation transforms (Etapa 4) -------------------------------------
// Mirror across the vertical axis through the data's bounding-box center.
TrackData mirrorX(const TrackData& base);
// Reverse driving direction (reverse waypoint order).
TrackData reverse(const TrackData& base);
// Scale the track width by `factor` (> 0).
TrackData scaleWidth(const TrackData& base, float factor);

// --- Procedural generation (Etapa 5) ---------------------------------------
// Deterministic for a given seed: same seed -> identical TrackData.
TrackData generateLoop(std::uint32_t seed, const GenParams& params);

} // namespace trackgen
