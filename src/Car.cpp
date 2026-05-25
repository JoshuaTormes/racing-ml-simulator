#include "Car.h"
#include "core/Constants.h"
#include <cmath>
#include <algorithm>

void Car::reset(Vec2 spawnPos, float spawnAngle) {
    pos             = spawnPos;
    angle           = spawnAngle;
    speed           = 0.f;
    fitness         = 0.f;
    maxProgress     = 0.f;
    idleTime        = 0.f;
    episodeTime     = 0.f;
    reversePenalty  = 0.f;
    regressPenalty  = 0.f;
    curvePenalty    = 0.f;
    speedBonus      = 0.f;
    noProgressTime  = 0.f;
    lowSpeedTime    = 0.f;
    done            = false;
    doneReason      = DoneReason::None;
    progState       = ProgressState{};
}

void Car::applyAction(const Action& a) {
    // Throttle: positive = accelerate, negative = brake
    float accelForce = (a.throttle > 0.f) ?  a.throttle * ACCEL * DT
                                           :  a.throttle * BRAKE * DT;
    // Negative throttle brakes; MAX_REVERSE_SPEED controls how far below zero speed can go
    speed += accelForce;
    speed *= DRAG;
    speed = std::clamp(speed, MAX_REVERSE_SPEED, MAX_SPEED);

    // Geometric limit scaled by speed (weak at low v, full at max v)
    // Grip limit kicks in at high speed: MAX_LAT_ACCEL/v < yawCmd → must brake before tight turn
    float yawCmd  = std::fabs(a.steering) * MAX_STEER * (std::fabs(speed) / MAX_SPEED);
    float yawGrip = MAX_LAT_ACCEL / std::max(std::fabs(speed), 1.f);
    float yawRate = std::copysign(std::min(yawCmd, yawGrip), a.steering);
    angle += yawRate * DT;

    // Integrate position
    pos.x += std::cos(angle) * speed * DT;
    pos.y += std::sin(angle) * speed * DT;
}

Observation Car::observe(const Track& track) const {
    // Sensor was already updated in stepDone; use its readings
    Observation obs{};
    for (int i = 0; i < NUM_RAYS; ++i)
        obs[i] = sensor.readings()[i];

    // speed normalized to [0,1]
    obs[NUM_RAYS] = std::fabs(speed) / MAX_SPEED;

    // angle to next waypoint, normalized to [-1,1]
    int n = (int)track.waypoints().size();
    int nextIdx  = progState.nextWp % n;
    int curIdx   = (progState.nextWp - 1 + n) % n;
    int next2Idx = track.closed() ? (progState.nextWp + 1) % n
                                  : std::min(progState.nextWp + 1, n - 1);

    Vec2 toWp = track.waypoints()[nextIdx] - pos;
    float wpAngle = std::atan2(toWp.y, toWp.x);
    float relAngle = wpAngle - angle;
    while (relAngle >  (float)M_PI) relAngle -= 2.f * (float)M_PI;
    while (relAngle < -(float)M_PI) relAngle += 2.f * (float)M_PI;
    obs[NUM_RAYS + 1] = relAngle / (float)M_PI;

    // distance to next waypoint normalized [0,1]
    float dist = toWp.length();
    obs[NUM_RAYS + 2] = std::min(dist / 400.f, 1.f);

    // angle to wp after next (lookahead), normalized to [-1,1]
    Vec2  toWp2      = track.waypoints()[next2Idx] - pos;
    float wp2Angle   = std::atan2(toWp2.y, toWp2.x);
    float relAngle2  = wp2Angle - angle;
    while (relAngle2 >  (float)M_PI) relAngle2 -= 2.f * (float)M_PI;
    while (relAngle2 < -(float)M_PI) relAngle2 += 2.f * (float)M_PI;
    obs[NUM_RAYS + 3] = relAngle2 / (float)M_PI;

    // signed curvature of the upcoming corner: angle between consecutive segments
    Vec2  seg1 = track.waypoints()[nextIdx]  - track.waypoints()[curIdx];
    Vec2  seg2 = track.waypoints()[next2Idx] - track.waypoints()[nextIdx];
    float cross = seg1.x * seg2.y - seg1.y * seg2.x;
    float dot   = seg1.x * seg2.x + seg1.y * seg2.y;
    obs[NUM_RAYS + 4] = std::atan2(cross, dot) / (float)M_PI;

    return obs;
}

float Car::stepDone(const Track& track, const RewardConfig& cfg) {
    if (done) return 0.f;

    float fitnessBefore = fitness;

    episodeTime += DT;

    // Update sensors
    sensor.update(pos, angle, track);

    // Update progress and high-water mark
    track.progressAt(pos, progState);
    if (progState.totalProg > maxProgress) {
        maxProgress     = progState.totalProg;
        noProgressTime  = 0.f; // reset stall counter whenever we advance
    } else {
        noProgressTime += DT;
    }

    // Penalty B: accumulate for each tick spent moving in reverse
    if (speed < 0.f)
        reversePenalty += cfg.w_reverse * (-speed) * DT;

    // Penalty C: accumulate for each tick spent behind the progress peak
    float regress = maxProgress - progState.totalProg;
    if (regress > 0.f)
        regressPenalty += cfg.w_regress * regress * DT;

    // Bonus F: reward maintaining speed while actively making forward progress
    if (progState.totalProg >= maxProgress)
        speedBonus += cfg.w_speed * (std::fabs(speed) / MAX_SPEED) * DT;

    // Penalty E: high speed through tight upcoming corners
    {
        int n2       = (int)track.waypoints().size();
        int curIdx   = (progState.nextWp - 1 + n2) % n2;
        int nextIdx  = progState.nextWp % n2;
        int next2Idx = track.closed() ? (progState.nextWp + 1) % n2
                                      : std::min(progState.nextWp + 1, n2 - 1);
        Vec2  seg1 = track.waypoints()[nextIdx]  - track.waypoints()[curIdx];
        Vec2  seg2 = track.waypoints()[next2Idx] - track.waypoints()[nextIdx];
        float cross    = seg1.x * seg2.y - seg1.y * seg2.x;
        float dot      = seg1.x * seg2.x + seg1.y * seg2.y;
        float curvature = std::fabs(std::atan2(cross, dot) / (float)M_PI);
        curvePenalty += cfg.w_curve * curvature * std::fabs(speed) * DT;
    }

    // Fitness: progress + speed bonus minus accumulated penalties (overwritten each tick)
    fitness = cfg.w_progress * maxProgress + speedBonus - reversePenalty - regressPenalty - curvePenalty;

    // Check done conditions
    int n        = (int)track.waypoints().size();
    int totalWps = n - 1;

    // Completed
    if (progState.nextWp > totalWps) {
        done = true; doneReason = DoneReason::Completed;
        fitness += cfg.w_finish + cfg.w_time * (EPISODE_TIMEOUT - episodeTime);
        return fitness - fitnessBefore;
    }
    // Timeout
    if (episodeTime >= EPISODE_TIMEOUT) {
        done = true; doneReason = DoneReason::Timeout;
        return fitness - fitnessBefore;
    }
    // Speed stall: catches cars physically stopped or barely moving (oscillation tricks the progress stall)
    if (std::fabs(speed) < STALL_SPEED) lowSpeedTime += DT;
    else                                lowSpeedTime  = 0.f;

    // Stall: no progress OR too slow for too long
    if (noProgressTime >= STALL_TIMEOUT || lowSpeedTime >= STALL_TIMEOUT) {
        done = true; doneReason = DoneReason::Stall;
        return fitness - fitnessBefore;
    }
    // Collision — check center + 4 corners of the car rectangle
    {
        float ca = std::cos(angle), sa = std::sin(angle);
        float hl = CAR_LENGTH * 0.5f, hw_c = CAR_WIDTH * 0.5f;
        Vec2 pts[5] = {
            pos,
            {pos.x + ca*hl - sa*hw_c, pos.y + sa*hl + ca*hw_c},
            {pos.x + ca*hl + sa*hw_c, pos.y + sa*hl - ca*hw_c},
            {pos.x - ca*hl - sa*hw_c, pos.y - sa*hl + ca*hw_c},
            {pos.x - ca*hl + sa*hw_c, pos.y - sa*hl - ca*hw_c},
        };
        bool outside = false;
        for (auto& p : pts) outside = outside || !track.isInsideTrack(p);
        if (outside) {
            done = true; doneReason = DoneReason::Collision;
            fitness -= cfg.w_crash;
            return fitness - fitnessBefore;
        }
    }

    return fitness - fitnessBefore;
}
