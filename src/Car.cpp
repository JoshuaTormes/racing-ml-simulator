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
    checkpointBonus = 0.f;
    lastNextWp      = 1;
    noProgressTime      = 0.f;
    progressAtLastReset = 0.f;
    lowSpeedTime        = 0.f;
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

    // Full steering authority at any speed; grip physics limits yaw at high speed naturally.
    // (Old formula scaled by speed/MAX_SPEED — made cornering physically impossible at safe speeds)
    float yawCmd  = std::fabs(a.steering) * MAX_STEER;
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

    // Pre-compute up to 6 sequential waypoint indices for 5-WP lookahead
    auto wpIdx = [&](int offset) -> int {
        return track.closed() ? (progState.nextWp + offset) % n
                              : std::min(progState.nextWp + offset, n - 1);
    };
    int i0 = curIdx, i1 = nextIdx, i2 = next2Idx;
    int i3 = wpIdx(2), i4 = wpIdx(3), i5 = wpIdx(4);

    // Helper: signed curvature (normalized) and speed_excess for a segment pair a→b→c
    auto curvAndExcess = [&](int a, int b, int c) -> std::pair<float,float> {
        Vec2  s1  = track.waypoints()[b] - track.waypoints()[a];
        Vec2  s2  = track.waypoints()[c] - track.waypoints()[b];
        float cr  = s1.x * s2.y - s1.y * s2.x;
        float dt  = s1.x * s2.x + s1.y * s2.y;
        float ang = std::atan2(cr, dt);
        float seg_len  = std::sqrt(s2.x * s2.x + s2.y * s2.y);
        float radius   = (std::fabs(ang) > 0.01f) ? seg_len / std::fabs(ang) : 9999.f;
        float safe_v   = std::min(std::sqrt(MAX_LAT_ACCEL * radius), MAX_SPEED);
        float excess   = std::max(-1.f, std::min(1.f, (std::fabs(speed) - safe_v) / MAX_SPEED));
        return { ang / (float)M_PI, excess };
    };

    // Lookaheads 1-5: each pair = (signed curvature, speed_excess)
    auto [c1, e1] = curvAndExcess(i0, i1, i2);  obs[NUM_RAYS+4]=c1; obs[NUM_RAYS+5]=e1;
    auto [c2, e2] = curvAndExcess(i1, i2, i3);  obs[NUM_RAYS+6]=c2; obs[NUM_RAYS+7]=e2;
    auto [c3, e3] = curvAndExcess(i2, i3, i4);  obs[NUM_RAYS+8]=c3; obs[NUM_RAYS+9]=e3;
    auto [c4, e4] = curvAndExcess(i3, i4, i5);  obs[NUM_RAYS+10]=c4; obs[NUM_RAYS+11]=e4;
    // 5th lookahead reuses last available index when near track end
    auto [c5, e5] = curvAndExcess(i4, i5, wpIdx(5)); obs[NUM_RAYS+12]=c5; obs[NUM_RAYS+13]=e5;

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
    if (progState.totalProg > maxProgress)
        maxProgress = progState.totalProg;

    // Reset stall timer only on meaningful advance (blocks spin-in-place oscillation)
    if (maxProgress >= progressAtLastReset + STALL_PROGRESS_MIN) {
        progressAtLastReset = maxProgress;
        noProgressTime      = 0.f;
    } else {
        noProgressTime += DT;
    }

    // Bonus G: one-shot reward for passing each curved waypoint
    if (progState.nextWp > lastNextWp) {
        int n2 = (int)track.waypoints().size();
        for (int wp = lastNextWp; wp < progState.nextWp; ++wp) {
            int prev = (wp - 1 + n2) % n2;
            int cur  =  wp           % n2;
            int next = (wp + 1)      % n2;
            Vec2  s1 = track.waypoints()[cur]  - track.waypoints()[prev];
            Vec2  s2 = track.waypoints()[next] - track.waypoints()[cur];
            float cr = s1.x * s2.y - s1.y * s2.x;
            float dt = s1.x * s2.x + s1.y * s2.y;
            float curv = std::fabs(std::atan2(cr, dt) / (float)M_PI);
            checkpointBonus += cfg.w_checkpoint * curv;
        }
        lastNextWp = progState.nextWp;
    }

    // Penalty B: accumulate for each tick spent moving in reverse
    if (speed < 0.f)
        reversePenalty += cfg.w_reverse * (-speed) * DT;

    // Penalty C: accumulate for each tick spent behind the progress peak
    float regress = maxProgress - progState.totalProg;
    if (regress > 0.f)
        regressPenalty += cfg.w_regress * regress * DT;

    // Compute upcoming curvature (shared by bonus F and penalty E)
    float curvatureAhead = 0.f;
    {
        int n2       = (int)track.waypoints().size();
        int curIdx   = (progState.nextWp - 1 + n2) % n2;
        int nextIdx  = progState.nextWp % n2;
        int next2Idx = track.closed() ? (progState.nextWp + 1) % n2
                                      : std::min(progState.nextWp + 1, n2 - 1);
        Vec2  seg1 = track.waypoints()[nextIdx]  - track.waypoints()[curIdx];
        Vec2  seg2 = track.waypoints()[next2Idx] - track.waypoints()[nextIdx];
        float cross = seg1.x * seg2.y - seg1.y * seg2.x;
        float dot   = seg1.x * seg2.x + seg1.y * seg2.y;
        curvatureAhead = std::fabs(std::atan2(cross, dot) / (float)M_PI);
    }

    // Bonus F: speed bonus only on straights — rewards re-acceleration out of corners
    // Threshold 0.15 ≈ ~27° turn angle; above that the road is considered curved.
    if (curvatureAhead < 0.15f && progState.totalProg >= maxProgress)
        speedBonus += cfg.w_speed * (std::fabs(speed) / MAX_SPEED) * DT;

    // Penalty E: high speed through tight corners (disabled by default, w_curve=0)
    curvePenalty += cfg.w_curve * curvatureAhead * std::fabs(speed) * DT;

    // Fitness: progress + bonuses minus accumulated penalties (overwritten each tick)
    fitness = cfg.w_progress * maxProgress + speedBonus + checkpointBonus - reversePenalty - regressPenalty - curvePenalty;

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
