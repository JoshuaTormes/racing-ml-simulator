#pragma once
#include "core/Vec2.h"
#include "core/Types.h"
#include "Track.h"
#include "Sensor.h"

enum class DoneReason { None, Collision, Timeout, Stall, Completed };

class Car {
public:
    // Public state (read by Game/Renderer)
    Vec2  pos             = {0, 0};
    float angle           = 0.f;
    float speed           = 0.f;
    float fitness         = 0.f;
    float maxProgress     = 0.f;  // high-water mark of progState.totalProg (monotonic)
    float idleTime        = 0.f;  // legacy: preserved for external readers, always 0
    float episodeTime     = 0.f;
    float reversePenalty  = 0.f;  // accumulated penalty for negative speed (task B)
    float regressPenalty  = 0.f;  // accumulated penalty for falling behind peak progress (task C)
    float curvePenalty    = 0.f;  // accumulated penalty for high speed through tight corners (task E)
    float speedBonus      = 0.f;  // accumulated bonus for maintaining speed while making progress (task F)
    float checkpointBonus = 0.f;  // one-shot bonus per curved waypoint passed (task G)
    int   lastNextWp      = 1;    // tracks nextWp to detect waypoint passage
    float noProgressTime      = 0.f;  // seconds since last meaningful progress (task D stall)
    float progressAtLastReset = 0.f;  // maxProgress value at last noProgressTime reset
    float lowSpeedTime        = 0.f;  // seconds below STALL_SPEED threshold
    bool  done            = false;
    DoneReason doneReason = DoneReason::None;
    ProgressState progState;
    Sensor sensor;

    void reset(Vec2 spawnPos, float spawnAngle);

    void applyAction(const Action& a);

    Observation observe(const Track& track) const;

    // Advance physics, evaluate done conditions; returns reward for this tick
    float stepDone(const Track& track, const RewardConfig& cfg);

};
