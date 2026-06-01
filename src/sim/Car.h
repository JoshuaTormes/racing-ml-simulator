#pragma once
#include "core/Vec2.h"
#include "core/Types.h"
#include "sim/Track.h"
#include "sim/Sensor.h"

enum class DoneReason { None, Collision, Timeout, Stall, Completed };

class Car {
public:
    // Public state (read by Game/Renderer)
    Vec2  pos             = {0, 0};
    float angle           = 0.f;
    float speed           = 0.f;
    float fitness         = 0.f;
    float maxProgress     = 0.f;  // high-water mark of progress relative to spawn ∈ [0,1]
    float spawnArcFrac    = 0.f;  // spawn position as fraction of total arc (0 = start line)
    float idleTime        = 0.f;  // legacy: preserved for external readers, always 0
    float episodeTime     = 0.f;
    float reversePenalty  = 0.f;
    float regressPenalty  = 0.f;
    float curvePenalty    = 0.f;
    float speedBonus      = 0.f;
    float checkpointBonus = 0.f;
    int   lastCheckpoint  = 1;    // next designWaypoint index to credit (0 = spawn; start past it)
    float noProgressTime      = 0.f;
    float progressAtLastReset = 0.f;
    float lowSpeedTime        = 0.f;
    bool  done            = false;
    DoneReason doneReason = DoneReason::None;
    ProjectionState projState;
    Sensor sensor;

    void reset(Vec2 spawnPos, float spawnAngle);
    // Spawn at an arbitrary point: spawnArcFrac ∈ [0,1) is the fraction of total arc
    // at the spawn, so progress in stepDone is measured relative to it.
    void reset(Vec2 spawnPos, float spawnAngle, float spawnArcFrac);

    void applyAction(const Action& a);

    Observation observe(const Track& track) const;

    // Advance physics, evaluate done conditions; returns reward for this tick
    float stepDone(const Track& track, const RewardConfig& cfg);

};
