#pragma once

constexpr int   SIM_HZ          = 60;
constexpr float DT              = 1.f / SIM_HZ;
constexpr int   NUM_RAYS        = 13;
constexpr float RAY_MAX_LEN     = 400.f;
constexpr float MAX_SPEED         = 400.f;
// Maximum reverse speed (px/s). 0 = reverse fully disabled (negative throttle acts as pure brake).
// Set to a small negative value (e.g. -40) to allow minor reversing to unstick the car. Range: [-MAX_SPEED, 0].
constexpr float MAX_REVERSE_SPEED = 0.f;
constexpr float ACCEL           = 300.f;
constexpr float BRAKE           = 500.f;
constexpr float DRAG            = 0.98f;
constexpr float MAX_STEER       = 3.0f;
constexpr float MAX_LAT_ACCEL   = 650.f;  // grip limit (px/s²): yawRate ≤ MAX_LAT_ACCEL/v
inline float EPISODE_TIMEOUT = 30.f;
inline void  setEpisodeTimeout(float t) { EPISODE_TIMEOUT = t; }
inline float episodeTimeout() noexcept  { return EPISODE_TIMEOUT; }
constexpr float STALL_TIMEOUT      = 2.f;
constexpr float STALL_SPEED        = 4.f;    // px/s — below this counts as stopped
// Min normalized arc progress to reset stall timer (progress ∈ [0,1] over centerline arc length).
// 0.003 ≈ 0.3% of the lap.
constexpr float STALL_PROGRESS_MIN = 0.003f;
// Centerline densification: Catmull-Rom sub-segments per original waypoint interval.
constexpr int   CENTERLINE_SUBSEGMENTS = 10;
// Arc window used by Track::curvatureAtArc (signed turn-angle over this much arc, in px).
// Re-used by Car::observe to derive a radius estimate from the angle.
constexpr float CURVATURE_ARC_WINDOW = 20.f;
// Lookahead distances (in centerline arc px) used by Car::observe for curvature inputs.
constexpr int   NUM_LOOKAHEADS    = 5;
constexpr float LOOKAHEAD_ARCS[NUM_LOOKAHEADS] = { 30.f, 60.f, 100.f, 160.f, 240.f };
// inputs: rays[0..NUM_RAYS-1], speed, lateral_offset, heading_error,
//         then NUM_LOOKAHEADS × (signed_curvature, speed_excess)
constexpr int   OBS_SIZE        = NUM_RAYS + 3 + 2 * NUM_LOOKAHEADS; // 26 with NUM_RAYS=13
inline int      NN_HIDDEN       = 32;
constexpr int   ACT_SIZE        = 2;
constexpr float CAR_LENGTH      = 20.f;
constexpr float CAR_WIDTH       = 10.f;
