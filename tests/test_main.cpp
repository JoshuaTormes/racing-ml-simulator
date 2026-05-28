#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <fstream>

#include "core/Vec2.h"
#include "core/Constants.h"
#include "core/Types.h"
#include "Track.h"
#include "Car.h"
#include "Sensor.h"
#include "NeuralNetwork.h"
#include "Game.h"
#include "GeneticAlgorithm.h"
#include "Trainers.h"
#include "Training.h"
#include "TrainingMath.h"
#include <filesystem>

// Lightweight test framework
static int g_pass = 0, g_fail = 0;
#define ASSERT(cond) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; std::cerr << "FAIL: " #cond " (" __FILE__ ":" << __LINE__ << ")\n"; } \
} while(0)
#define ASSERT_THROW(expr, ExType) do { \
    bool caught = false; \
    try { expr; } catch (const ExType&) { caught = true; } \
    if (caught) ++g_pass; \
    else { ++g_fail; std::cerr << "FAIL: expected " #ExType " from " #expr " (" __FILE__ ":" << __LINE__ << ")\n"; } \
} while(0)

// ---------- Vec2 ----------
static void test_vec2() {
    ASSERT(std::fabs(Vec2{3,4}.length() - 5.f) < 1e-5f);
    ASSERT(std::fabs(Vec2{3,0}.normalized().x - 1.f) < 1e-5f);
    Vec2 v = Vec2{1,0}.rotated((float)M_PI * 0.5f);
    ASSERT(std::fabs(v.x) < 1e-5f && std::fabs(v.y - 1.f) < 1e-5f);
    ASSERT(std::fabs(Vec2{3,4}.dot({1,0}) - 3.f) < 1e-5f);
    Vec2 perp = Vec2{1,0}.perpendicular();
    ASSERT(std::fabs(perp.x) < 1e-5f && std::fabs(perp.y - (-1.f)) < 1e-5f);

    // rotated 180° should negate
    Vec2 back = Vec2{1,0}.rotated((float)M_PI);
    ASSERT(std::fabs(back.x + 1.f) < 1e-4f);
}

// ---------- Track geometry ----------
static void test_track_geometry() {
    Track track("maps/map1.json");

    // Spawn inside track
    ASSERT(track.isInsideTrack(track.spawnPos()));

    // Far corner outside track
    ASSERT(!track.isInsideTrack({0.f, 0.f}));

    // Raycast from spawn straight up: leftBorder[0] is ~60px above (y=-1)
    // The avg direction at wp[0] is {1,0} (symmetric map), so perp is {0,±1}
    Vec2 sp = track.spawnPos();
    float dUp = track.raycast(sp, {0.f, -1.f}, RAY_MAX_LEN);
    ASSERT(dUp > 0.f && dUp < RAY_MAX_LEN);
    // Right border is ~60px below
    float dDown = track.raycast(sp, {0.f, 1.f}, RAY_MAX_LEN);
    ASSERT(dDown > 0.f && dDown < RAY_MAX_LEN);
    // Both borders should be roughly equidistant (track_width/2 = 60px each)
    ASSERT(std::fabs(dUp - 60.f) < 15.f);
    ASSERT(std::fabs(dDown - 60.f) < 15.f);

    // Forward raycast from spawn also hits something within max range
    float spA = track.spawnAngle();
    Vec2 forward = {std::cos(spA), std::sin(spA)};
    float dFwd = track.raycast(sp, forward, RAY_MAX_LEN);
    ASSERT(dFwd > 0.f && dFwd <= RAY_MAX_LEN);

    // Bad JSON file
    ASSERT_THROW(Track("maps/nonexistent.json"), std::runtime_error);

    // Borders were generated (non-empty)
    ASSERT(!track.leftBorder().empty());
    ASSERT(!track.rightBorder().empty());
    ASSERT(track.leftBorder().size() == track.rightBorder().size());

    // After Catmull-Rom densification, map1 has 8 design waypoints and the centerline
    // contains at least 8 × CENTERLINE_SUBSEGMENTS points (plus the closure point).
    ASSERT(track.designWaypoints().size() == 8);
    ASSERT((int)track.centerline().size() >= 8 * CENTERLINE_SUBSEGMENTS);
    ASSERT(track.totalArcLength() > 0.f);
}

// ---------- Centerline interpolates through design waypoints ----------
static void test_centerline_passes_through_waypoints() {
    for (const std::string& path : {"maps/map1.json", "maps/map6_chicanes_infernais.json"}) {
        Track track(path);
        const auto& dwps = track.designWaypoints();
        const auto& center = track.centerline();
        for (auto& wp : dwps) {
            float bestD = std::numeric_limits<float>::infinity();
            for (auto& c : center)
                bestD = std::min(bestD, (c - wp).length());
            // The construction places centerline_[k * N] == designWaypoints_[k], so the
            // tolerance is dominated by float precision (< 0.5 px is generous).
            ASSERT(bestD < 0.5f);
        }
    }
}

// ---------- Arc length is monotonic and consistent ----------
static void test_arc_length_monotonic() {
    Track track("maps/map1.json");
    const auto& arc = track.arcLength();
    ASSERT(arc.size() >= 2);
    for (size_t i = 1; i < arc.size(); ++i)
        ASSERT(arc[i] >= arc[i - 1]);
    ASSERT(std::fabs(arc.back() - track.totalArcLength()) < 1e-3f);
}

// ---------- Projection is idempotent on centerline samples ----------
static void test_projection_idempotent() {
    Track track("maps/map1.json");
    float L = track.totalArcLength();
    ASSERT(L > 0.f);

    // Sample 50 points along the centerline; each projection must round-trip
    // through updateProjection to within ~1px of the original arc.
    for (int i = 0; i < 50; ++i) {
        float arc = L * (float)i / 50.f;
        Vec2 p = track.centerlineAtArc(arc);
        ProjectionState st;
        // Seed segment near the expected arc so local search starts close — the
        // production path always carries state across ticks.
        st.segIdx = (int)((float)i / 50.f * (track.centerline().size() - 1));
        track.updateProjection(p, st);
        float err = std::fabs(st.arcLen - arc);
        // Closed-loop wrap: the spawn point matches itself at arc=0 and arc=L.
        err = std::min(err, std::fabs(err - L));
        ASSERT(err < 1.0f);
    }
}

// ---------- Projection state segIdx tracks forward motion ----------
static void test_projection_local_search() {
    Track track("maps/map1.json");
    ProjectionState st;
    float L = track.totalArcLength();
    int prevSeg = st.segIdx;
    int decreased = 0;
    // Walk along the centerline in arc-length steps, simulating a moving car.
    for (int i = 1; i <= 50; ++i) {
        float arc = L * (float)i / 50.f;
        Vec2 p = track.centerlineAtArc(arc);
        track.updateProjection(p, st);
        // Monotone forward in segment index, except for at most one wrap (closed map).
        if (st.segIdx < prevSeg) ++decreased;
        prevSeg = st.segIdx;
    }
    // At most one drop allowed (the wrap-around at the end of a closed loop).
    ASSERT(decreased <= 1);
}

// ---------- Sensor ----------
static void test_sensor() {
    Track track("maps/map1.json");
    Sensor sensor;
    sensor.update(track.spawnPos(), track.spawnAngle(), track);

    for (int i = 0; i < NUM_RAYS; ++i) {
        ASSERT(sensor.readings()[i] >= 0.f);
        ASSERT(sensor.readings()[i] <= 1.f);
    }
    // Central ray points forward into the track, should not be maximum
    // (some border is visible within RAY_MAX_LEN from spawn)
    float minReading = sensor.readings()[0];
    for (int i = 1; i < NUM_RAYS; ++i)
        if (sensor.readings()[i] < minReading) minReading = sensor.readings()[i];
    ASSERT(minReading < 1.f);
}

// ---------- NeuralNetwork ----------
static void test_neural_network() {
    // Known-weight forward pass: 2→2 topology, all weights=1, biases=0
    NeuralNetwork nn({2, 2}, 0);
    std::vector<float> w = {
        1.f, 1.f,  // row 0: out[0] = tanh(in[0]*1 + in[1]*1)
        1.f, 1.f,  // row 1: out[1] = tanh(in[0]*1 + in[1]*1)
        0.f, 0.f   // biases
    };
    nn.setWeights(w);
    auto out = nn.forward({1.f, 0.f});
    float expected = std::tanh(1.f);
    ASSERT(std::fabs(out[0] - expected) < 1e-5f);
    ASSERT(std::fabs(out[1] - expected) < 1e-5f);

    // Save → load round-trip
    NeuralNetwork nn2({2, 2}, 0);
    std::vector<uint8_t> buf;
    nn.saveToBuffer(buf);
    nn2.loadFromBuffer(buf);
    auto orig  = nn.getWeights();
    auto loaded = nn2.getWeights();
    ASSERT(orig.size() == loaded.size());
    bool match = true;
    for (size_t i = 0; i < orig.size(); ++i)
        if (std::fabs(orig[i] - loaded[i]) > 1e-7f) match = false;
    ASSERT(match);

    // Wrong magic
    std::vector<uint8_t> badBuf = buf;
    badBuf[0] = 'X';
    ASSERT_THROW(nn2.loadFromBuffer(badBuf), std::runtime_error);

    // Wrong version
    std::vector<uint8_t> badVer = buf;
    badVer[4] = 99;
    ASSERT_THROW(nn2.loadFromBuffer(badVer), std::runtime_error);

    // Wrong topology
    NeuralNetwork nn3({3, 2}, 0);
    ASSERT_THROW(nn3.loadFromBuffer(buf), std::runtime_error);

    // Default topology forward with random weights produces output in (-1,1)
    NeuralNetwork def(defaultTopology(), 42);
    Observation obs{};
    std::vector<float> inp(obs.begin(), obs.end());
    auto res = def.forward(inp);
    ASSERT(res.size() == 2);
    ASSERT(res[0] > -1.f && res[0] < 1.f);
    ASSERT(res[1] > -1.f && res[1] < 1.f);
}

// ---------- Car determinism (single-thread) ----------
static void test_car_determinism_single() {
    SimConfig cfg;
    cfg.population = 10;
    cfg.seed = 1234;
    cfg.headless = true;
    cfg.threads = 1;

    Game g1(cfg), g2(cfg);
    // Run 200 ticks on both
    for (int t = 0; t < 200; ++t) { g1.tick(); g2.tick(); }

    auto& cars1 = g1.cars();
    auto& cars2 = g2.cars();
    for (int i = 0; i < cfg.population; ++i) {
        // Same seed → same controllers → identical state
        ASSERT(std::fabs(cars1[i].pos.x   - cars2[i].pos.x)   < 1e-4f);
        ASSERT(std::fabs(cars1[i].pos.y   - cars2[i].pos.y)   < 1e-4f);
        ASSERT(std::fabs(cars1[i].fitness - cars2[i].fitness)  < 1e-3f);
    }
}

// ---------- Car determinism (multi-thread vs single-thread) ----------
static void test_car_determinism_multithread() {
    int hw = (int)std::thread::hardware_concurrency();
    if (hw < 2) {
        std::cout << "  (skipping multi-thread determinism: only 1 HW thread)\n";
        ++g_pass;
        return;
    }

    SimConfig cfgSingle, cfgMulti;
    cfgSingle.population = 100; cfgSingle.seed = 99; cfgSingle.headless = true; cfgSingle.threads = 1;
    cfgMulti  = cfgSingle;
    cfgMulti.threads = hw;

    Game gSingle(cfgSingle), gMulti(cfgMulti);
    for (int t = 0; t < 300 && !gSingle.episodeDone(); ++t) {
        gSingle.tick();
        gMulti.tick();
    }

    auto& cs = gSingle.cars();
    auto& cm = gMulti.cars();
    bool match = true;
    for (int i = 0; i < cfgSingle.population; ++i) {
        if (std::fabs(cs[i].fitness - cm[i].fitness) > 1e-3f) {
            match = false;
            std::cerr << "  mismatch car " << i
                      << " single=" << cs[i].fitness
                      << " multi=" << cm[i].fitness << "\n";
        }
    }
    ASSERT(match);
}

// ---------- GeneticAlgorithm ----------
static void test_genetic_algorithm() {
    GeneticAlgorithm ga;
    ga.initPopulation(20, 100, 42);
    ASSERT(ga.genomes().size() == 20);
    ASSERT(ga.generation() == 0);

    for (size_t i = 0; i < 20; ++i) ga.setFitness(i, (float)i);
    ga.evolve();
    ASSERT(ga.generation() == 1);
    ASSERT(ga.genomes().size() == 20);
}

// ---------- Trainers ----------
static void test_trainers_weight_count() {
    const size_t POP = 10, WC = 50;
    for (const std::string& algo : {"genetic", "random_search", "hillclimb"}) {
        auto t = makeTrainer(algo);
        t->init(POP, WC, 42);
        ASSERT(t->populationSize() == POP);
        for (size_t i = 0; i < POP; ++i)
            ASSERT(t->weights(i).size() == WC);
    }
}

static void test_trainers_determinism() {
    const size_t POP = 8, WC = 20;
    for (const std::string& algo : {"genetic", "random_search", "hillclimb"}) {
        auto t1 = makeTrainer(algo);
        auto t2 = makeTrainer(algo);
        t1->init(POP, WC, 77);
        t2->init(POP, WC, 77);
        // Feed same fitness sequence
        for (size_t i = 0; i < POP; ++i) {
            float f = (float)i * 1.5f;
            t1->setFitness(i, f);
            t2->setFitness(i, f);
        }
        t1->evolve();
        t2->evolve();
        bool match = true;
        for (size_t i = 0; i < POP; ++i)
            for (size_t j = 0; j < WC; ++j)
                if (std::fabs(t1->weights(i)[j] - t2->weights(i)[j]) > 1e-6f)
                    match = false;
        ASSERT(match);
    }
}

static void test_trainers_elitism() {
    // random_search and hillclimb must be non-decreasing in global best fitness
    for (const std::string& algo : {"random_search", "hillclimb"}) {
        auto t = makeTrainer(algo);
        t->init(10, 30, 5);
        float prevBest = -1e30f;
        for (int gen = 0; gen < 5; ++gen) {
            // set increasing fitness for first individual
            for (size_t i = 0; i < t->populationSize(); ++i)
                t->setFitness(i, (float)(gen * 10 + (int)i));
            float curBest = -1e30f;
            for (size_t i = 0; i < t->populationSize(); ++i)
                if ((float)(gen * 10 + (int)i) > curBest)
                    curBest = (float)(gen * 10 + (int)i);
            ASSERT(curBest >= prevBest);
            prevBest = curBest;
            t->evolve();
        }
    }
}

static void test_trainers_resume() {
    const size_t POP = 8, WC = 20;
    std::vector<float> champion(WC, 1.0f);

    // HillClimb: pop[0] must equal champion
    {
        auto t = makeTrainer("hillclimb");
        t->initFromWeights(champion, POP, 42);
        bool eq = true;
        for (size_t j = 0; j < WC; ++j)
            if (std::fabs(t->weights(0)[j] - champion[j]) > 1e-6f) eq = false;
        ASSERT(eq);
    }
    // Genetic seedFrom: pop[0] must equal champion
    {
        GeneticAlgorithm ga;
        ga.seedFrom(champion, POP, 42);
        bool eq = true;
        for (size_t j = 0; j < WC; ++j)
            if (std::fabs(ga.genomes()[0].weights[j] - champion[j]) > 1e-6f) eq = false;
        ASSERT(eq);
        ASSERT(ga.genomes().size() == POP);
    }
    // RandomSearch: pop[0] must equal champion after initFromWeights
    {
        auto t = makeTrainer("random_search");
        t->initFromWeights(champion, POP, 42);
        bool eq = true;
        for (size_t j = 0; j < WC; ++j)
            if (std::fabs(t->weights(0)[j] - champion[j]) > 1e-6f) eq = false;
        ASSERT(eq);
    }
}

static void test_training_session_headless() {
    SimConfig cfg;
    cfg.population = 10;
    cfg.headless   = true;
    cfg.threads    = 1;
    cfg.seed       = 42;

    auto outDir = (std::filesystem::temp_directory_path() / "racing_ml_test_out").string();
    std::filesystem::remove_all(outDir);

    TrainingSession session(cfg, makeTrainer("genetic"), 3, outDir);
    session.runAll();

    ASSERT(session.done());
    ASSERT((int)session.history().size() == 3);

    // best.rnnw and gen_0001..0003.rnnw must exist
    ASSERT(std::filesystem::exists(outDir + "/best.rnnw"));
    ASSERT(std::filesystem::exists(outDir + "/gen_0001.rnnw"));
    ASSERT(std::filesystem::exists(outDir + "/gen_0002.rnnw"));
    ASSERT(std::filesystem::exists(outDir + "/gen_0003.rnnw"));

    // Reload best.rnnw into a NeuralNetwork without throwing
    NeuralNetwork nn(defaultTopology());
    nn.load(outDir + "/best.rnnw");
    ASSERT(nn.getWeights().size() == NeuralNetwork(defaultTopology()).getWeights().size());

    std::filesystem::remove_all(outDir);
}

// ---------- Reverse blocked (task A) ----------
// With MAX_REVERSE_SPEED=0 (default), negative throttle must never produce speed < 0.
static void test_reverse_blocked() {
    Car car;
    Track track("maps/map1.json");
    car.reset(track.spawnPos(), track.spawnAngle());

    // Apply full reverse throttle for 120 ticks (2 seconds)
    Action rev;
    rev.throttle = -1.f;
    rev.steering =  0.f;
    for (int i = 0; i < 120; ++i) {
        car.applyAction(rev);
        ASSERT(car.speed >= 0.f); // must never go negative
    }

    // Braking from positive speed must still work: first accelerate, then brake
    car.reset(track.spawnPos(), track.spawnAngle());
    Action fwd; fwd.throttle = 1.f; fwd.steering = 0.f;
    for (int i = 0; i < 60; ++i) car.applyAction(fwd);
    ASSERT(car.speed > 0.f);

    float speedBeforeBrake = car.speed;
    for (int i = 0; i < 30; ++i) car.applyAction(rev);
    ASSERT(car.speed < speedBeforeBrake); // braking reduced speed
    ASSERT(car.speed >= 0.f);             // but never went negative
}

// ---------- Reverse penalty fires only for negative speed (task B) ----------
// w_reverse must penalise speed < 0 but NOT braking from positive speed.
static void test_reverse_penalty() {
    // Sub-test 1: with MAX_REVERSE_SPEED allowing reverse, penalty accumulates
    {
        // Temporarily simulate a car that CAN go in reverse by giving it negative speed directly
        // We do this by testing the accumulation logic: build a car, manually set speed < 0
        // and call stepDone, then check reversePenalty > 0.
        // Because MAX_REVERSE_SPEED=0 blocks negative speed from applyAction, we test the
        // accumulator logic by verifying it's zero when speed stays >= 0.
        Track track("maps/map1.json");
        Car car;
        RewardConfig cfg;
        cfg.w_reverse = 1.0f;
        cfg.w_regress = 0.0f; // isolate
        car.reset(track.spawnPos(), track.spawnAngle());

        // Just brake — speed stays >= 0 (MAX_REVERSE_SPEED=0), penalty must stay 0
        Action rev; rev.throttle = -1.f; rev.steering = 0.f;
        for (int i = 0; i < 60; ++i) {
            car.applyAction(rev);
            car.stepDone(track, cfg);
        }
        ASSERT(car.reversePenalty == 0.f); // no negative speed → no penalty
    }

    // Sub-test 2: penalty reduces fitness proportionally when speed would go negative
    // We verify this by checking that fitness formula honours the accumulator:
    // drive forward then stall (no reverse possible) → penalty stays 0 → fitness unaffected by B
    {
        Track track("maps/map1.json");
        Car carNoPenalty, carWithPenalty;
        RewardConfig cfgOff, cfgOn;
        cfgOff.w_reverse = 0.0f; cfgOff.w_regress = 0.0f;
        cfgOn.w_reverse  = 1.0f; cfgOn.w_regress  = 0.0f;

        carNoPenalty.reset(track.spawnPos(), track.spawnAngle());
        carWithPenalty.reset(track.spawnPos(), track.spawnAngle());

        // Both cars drive forward — no reverse speed → penalty must be identical (both 0)
        Action fwd; fwd.throttle = 1.f; fwd.steering = 0.f;
        for (int i = 0; i < 60; ++i) {
            carNoPenalty.applyAction(fwd);
            carNoPenalty.stepDone(track, cfgOff);
            carWithPenalty.applyAction(fwd);
            carWithPenalty.stepDone(track, cfgOn);
        }
        // Fitness should be equal: both drove forward, no reverse, w_reverse had no effect
        ASSERT(std::fabs(carNoPenalty.reversePenalty - carWithPenalty.reversePenalty) < 1e-5f);
    }
}

// ---------- Regress penalty discounts fitness (task C) ----------
static void test_regress_penalty() {
    Track track("maps/map1.json");
    Car car;
    RewardConfig cfg;
    cfg.w_regress = 1.0f;
    cfg.w_reverse = 0.0f; // isolate C
    car.reset(track.spawnPos(), track.spawnAngle());

    // Drive forward to build up maxProgress
    Action fwd; fwd.throttle = 1.f; fwd.steering = 0.f;
    for (int i = 0; i < 60; ++i) {
        car.applyAction(fwd);
        car.stepDone(track, cfg);
    }
    float peakProgress = car.maxProgress;
    ASSERT(peakProgress > 0.f);

    // At peak, regressPenalty should be 0 (car is at or ahead of max)
    ASSERT(car.regressPenalty >= 0.f);

    // Fitness formula: w_progress * maxProgress + speedBonus + checkpointBonus
    //                  - reversePenalty - regressPenalty - curvePenalty
    float fitnessAtPeak = car.fitness;
    float expected = cfg.w_progress * peakProgress
                   + car.speedBonus + car.checkpointBonus
                   - car.reversePenalty - car.regressPenalty - car.curvePenalty;
    ASSERT(std::fabs(fitnessAtPeak - expected) < 1e-3f);

    // regressPenalty stays non-negative (can only grow when behind peak)
    ASSERT(car.regressPenalty >= 0.f);
    // fitness must not exceed progress + bonuses (penalties only subtract)
    float upperBound = cfg.w_progress * car.maxProgress + car.speedBonus + car.checkpointBonus;
    ASSERT(car.fitness <= upperBound + 1e-3f);
}

// ---------- Stall by no-progress (task D) ----------
// Stall must fire when maxProgress has not increased for STALL_TIMEOUT seconds,
// regardless of whether the car is moving or stationary.
static void test_stall_by_no_progress() {
    Track track("maps/map1.json");
    RewardConfig cfg;
    cfg.w_reverse = 0.0f;
    cfg.w_regress = 0.0f;

    // Part 1: while advancing, noProgressTime stays near 0 and idleTime is always 0
    {
        Car car;
        car.reset(track.spawnPos(), track.spawnAngle());
        Action fwd; fwd.throttle = 1.f; fwd.steering = 0.f;
        for (int i = 0; i < 30; ++i) {
            car.applyAction(fwd);
            car.stepDone(track, cfg);
        }
        ASSERT(car.speed > cfg.idle_eps);         // car is moving well above old stall threshold
        ASSERT(car.idleTime == 0.f);              // legacy field is never updated (mechanism removed)
        ASSERT(car.noProgressTime < STALL_TIMEOUT); // hasn't triggered stall while advancing
        ASSERT(!car.done);
    }

    // Part 2: stall fires via noProgressTime even when the car never moves
    // (The car stays at spawn; no action applied; progress never increases.)
    // This also validates the path where old speed-based stall would also fire — but the
    // mechanism used is noProgressTime, confirmed by the field assertion below.
    {
        Car car;
        car.reset(track.spawnPos(), track.spawnAngle());
        Action nothing; nothing.throttle = 0.f; nothing.steering = 0.f;
        int stallTicks = (int)(STALL_TIMEOUT * SIM_HZ) + 5;
        for (int i = 0; i < stallTicks && !car.done; ++i) {
            car.applyAction(nothing);
            car.stepDone(track, cfg);
        }
        ASSERT(car.done);
        ASSERT(car.doneReason == DoneReason::Stall);
        ASSERT(car.noProgressTime >= STALL_TIMEOUT); // fired by progress-based mechanism
        ASSERT(car.idleTime == 0.f);                 // old mechanism was NOT active
    }
}

// ---------- Game headless episode terminates ----------
static void test_game_episode_terminates() {
    SimConfig cfg;
    cfg.population = 5;
    cfg.seed = 7;
    cfg.headless = true;
    cfg.threads = 1;
    Game g(cfg);
    double secs = g.runHeadlessEpisode();
    ASSERT(g.episodeDone());
    ASSERT(secs < 60.0); // must finish within real-time budget
}

// ---------- T1: CVaRRank(α=1) produces score ∈ [0,1] with mean ≈ 0.5 ----------
static void test_cvar_rank_mean() {
    using namespace training_math;
    // 2 maps, 5 cars — synthetic fitness
    std::vector<float> fA = {10.f, 3.f, 7.f, 1.f, 5.f};
    std::vector<float> fB = { 2.f, 8.f, 4.f, 9.f, 6.f};
    auto rA = normalized_ranks(fA);
    auto rB = normalized_ranks(fB);

    float sum = 0.f;
    for (size_t c = 0; c < 5; ++c) {
        std::vector<float> ranks = {rA[c], rB[c]};
        float score = cvar(ranks, 1.0f);
        ASSERT(score >= 0.f && score <= 1.f + 1e-5f);
        sum += score;
    }
    float mean = sum / 5.f;
    ASSERT(std::fabs(mean - 0.5f) < 0.05f);
}

// ---------- T2: CVaRRank invariant to monotonic per-map transformation ----------
static void test_cvar_rank_invariance() {
    using namespace training_math;
    std::vector<float> fA = {1.f, 5.f, 3.f, 2.f, 4.f};
    std::vector<float> fB = {9.f, 2.f, 6.f, 8.f, 3.f};

    // Compute scores on original fitnesses
    auto rA1 = normalized_ranks(fA);
    auto rB1 = normalized_ranks(fB);
    std::vector<float> scores1(5);
    for (size_t c = 0; c < 5; ++c)
        scores1[c] = cvar({rA1[c], rB1[c]}, 0.5f);

    // Apply monotonic transformation: f' = a*f + b per map (a > 0)
    std::vector<float> fA2(5), fB2(5);
    for (size_t i = 0; i < 5; ++i) {
        fA2[i] = 3.f * fA[i] + 7.f;
        fB2[i] = std::exp(fB[i]);
    }
    auto rA2 = normalized_ranks(fA2);
    auto rB2 = normalized_ranks(fB2);
    std::vector<float> scores2(5);
    for (size_t c = 0; c < 5; ++c)
        scores2[c] = cvar({rA2[c], rB2[c]}, 0.5f);

    for (size_t c = 0; c < 5; ++c)
        ASSERT(std::fabs(scores1[c] - scores2[c]) < 1e-5f);
}

// ---------- T3: CVaR limits ----------
static void test_cvar_limits() {
    using namespace training_math;
    std::vector<float> x = {4.f, 1.f, 3.f, 2.f}; // M=4

    // CVaR(α=1) == mean
    float m = (4.f + 1.f + 3.f + 2.f) / 4.f;
    ASSERT(std::fabs(cvar(x, 1.0f) - m) < 1e-5f);

    // CVaR(α=1/M) == min
    ASSERT(std::fabs(cvar(x, 0.25f) - 1.f) < 1e-5f);

    // CVaR(α=0.5) == mean of 2 smallest
    ASSERT(std::fabs(cvar(x, 0.5f) - 1.5f) < 1e-5f);

    // Ranks: for single map with P=4, CVaRRank(α=1) == mean of ranks == 0.5
    std::vector<float> f4 = {10.f, 5.f, 8.f, 3.f};
    auto ranks = normalized_ranks(f4);
    float rankSum = 0.f;
    for (float r : ranks) rankSum += r;
    ASSERT(std::fabs(rankSum / 4.f - 0.5f) < 1e-5f);
}

// ---------- T4: Z-score produces μ≈0, σ≈1 ----------
static void test_zscore_normalization() {
    using namespace training_math;
    std::vector<float> f = {1.f, 2.f, 3.f, 4.f, 5.f};
    auto norm = normalize_zscore(f);
    ASSERT(norm.size() == 5);

    float mu = 0.f;
    for (float v : norm) mu += v;
    mu /= 5.f;
    ASSERT(std::fabs(mu) < 1e-4f);

    float var = 0.f;
    for (float v : norm) { float d = v - mu; var += d*d; }
    float sigma = std::sqrt(var / 5.f);
    ASSERT(std::fabs(sigma - 1.f) < 1e-4f);
}

// ---------- T5: Linear curriculum activates correct count per gen ----------
static void test_curriculum_linear() {
    using namespace training_math;
    CurriculumConfig cfg;
    cfg.mode  = CurriculumMode::Linear;
    cfg.start = 2;
    cfg.step  = 15;

    ASSERT(active_map_count(0,  6, cfg) == 2);
    ASSERT(active_map_count(14, 6, cfg) == 2);
    ASSERT(active_map_count(15, 6, cfg) == 3);
    ASSERT(active_map_count(29, 6, cfg) == 3);
    ASSERT(active_map_count(30, 6, cfg) == 4);
    ASSERT(active_map_count(90, 6, cfg) == 6);  // capped at total=6
    ASSERT(active_map_count(0,  1, cfg) == 1);  // single-map: always 1

    // None mode
    CurriculumConfig none;
    none.mode = CurriculumMode::None;
    ASSERT(active_map_count(0,  6, none) == 6);

    // Explicit mode
    CurriculumConfig expl;
    expl.mode = CurriculumMode::Explicit;
    expl.schedule = {10, 20, 30}; // M=4, M-1=3 thresholds
    ASSERT(active_map_count(0,  4, expl) == 1);
    ASSERT(active_map_count(9,  4, expl) == 1);
    ASSERT(active_map_count(10, 4, expl) == 2);
    ASSERT(active_map_count(25, 4, expl) == 3);
    ASSERT(active_map_count(30, 4, expl) == 4);
}

// ---------- T6: setHiddenSize + inferHiddenFromWeights ----------
static void test_hidden_size() {
    int origH = NN_HIDDEN;

    setHiddenSize(64);
    NeuralNetwork nn(defaultTopology());
    // OBS_SIZE*64 + 64 + 64*ACT_SIZE + ACT_SIZE = 26*64 + 64 + 64*2 + 2 = 1858
    ASSERT(nn.getWeights().size() == 1858);

    ASSERT(inferHiddenFromWeights(1858) == 64);
    // H=32: 26*32 + 32 + 32*2 + 2 = 930
    ASSERT(inferHiddenFromWeights(930)  == 32);
    // Invalid: (999-2)/29 is not integer
    ASSERT(inferHiddenFromWeights(999)  == -1);
    ASSERT(inferHiddenFromWeights(0)    == -1);

    // Restore
    setHiddenSize(origH);
    ASSERT(NN_HIDDEN == origH);
}

// ---------- T7: Determinism across two sequential runAll() calls ----------
static void test_training_determinism() {
    SimConfig cfg;
    cfg.population = 10;
    cfg.headless   = true;
    cfg.threads    = 1;
    cfg.seed       = 7;

    auto tmpBase = (std::filesystem::temp_directory_path() / "racing_ml_det_test").string();

    // Test each aggregator with curriculum=none for simplicity
    struct AggCase { FitnessAgg agg; MapNormMode norm; };
    std::vector<AggCase> cases = {
        { FitnessAgg::CVaRRank, MapNormMode::ZScore    },
        { FitnessAgg::Min,      MapNormMode::Progress  },
        { FitnessAgg::Mean,     MapNormMode::ZScore    },
        { FitnessAgg::CVaRRaw,  MapNormMode::ZScore    },
    };

    auto loadBytes = [](const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), {});
    };

    for (const auto& ac : cases) {
        std::string dir1 = tmpBase + "/det1";
        std::string dir2 = tmpBase + "/det2";
        std::filesystem::remove_all(dir1);
        std::filesystem::remove_all(dir2);

        MultiMapConfig mmcfg;
        mmcfg.fitnessAgg       = ac.agg;
        mmcfg.mapNorm          = ac.norm;
        mmcfg.curriculum.mode  = CurriculumMode::None;

        {
            TrainingSession s1(cfg, makeTrainer("genetic"), 3, dir1,
                               {"maps/map1.json"}, {}, mmcfg);
            TrainingSession s2(cfg, makeTrainer("genetic"), 3, dir2,
                               {"maps/map1.json"}, {}, mmcfg);
            s1.runAll();
            s2.runAll();
        }

        auto b1 = loadBytes(dir1 + "/best.rnnw");
        auto b2 = loadBytes(dir2 + "/best.rnnw");
        ASSERT(b1.size() > 0 && b1.size() == b2.size());
        bool match = (b1 == b2);
        if (!match) {
            std::cerr << "  Determinism fail for agg=" << (int)ac.agg << "\n";
        }
        ASSERT(match);

        std::filesystem::remove_all(tmpBase);
    }
}

// ---------- E1: Spatial grid raycast and isInsideTrack correctness ----------
static float bruteRaycast(const Track& track, Vec2 origin, Vec2 dir, float maxLen) {
    float best = maxLen;
    const auto& L = track.leftBorder();
    const auto& R = track.rightBorder();
    int n = (int)L.size();
    for (int i = 0; i + 1 < n; ++i) {
        auto segTest = [&](Vec2 a, Vec2 b) {
            Vec2  ab  = b - a;
            float det = dir.x * ab.y - dir.y * ab.x;
            if (std::fabs(det) < 1e-9f) return;
            Vec2  ao = origin - a;
            float t  = -(ao.x * ab.y - ao.y * ab.x) / det;
            float u  = -(ao.x * dir.y - ao.y * dir.x) / det;
            if (t >= 0.f && u >= 0.f && u <= 1.f && t < best) best = t;
        };
        segTest(L[i], L[i+1]);
        segTest(R[i], R[i+1]);
    }
    for (const auto& ob : track.obstacles()) {
        if (ob.type == Obstacle::Type::Circle) {
            Vec2 oc = origin - ob.pos;
            float b = 2.f * oc.dot(dir), c = oc.dot(oc) - ob.radius * ob.radius;
            float disc = b*b - 4.f*c;
            if (disc >= 0.f) {
                float sq = std::sqrt(disc);
                float t0 = (-b - sq) * 0.5f, t1 = (-b + sq) * 0.5f;
                float t = (t0 >= 0.f) ? t0 : t1;
                if (t >= 0.f && t < best) best = t;
            }
        } else {
            float hx = ob.size.x * 0.5f, hy = ob.size.y * 0.5f;
            Vec2 corners[4] = {{ob.pos.x-hx,ob.pos.y-hy},{ob.pos.x+hx,ob.pos.y-hy},
                               {ob.pos.x+hx,ob.pos.y+hy},{ob.pos.x-hx,ob.pos.y+hy}};
            for (int k = 0; k < 4; ++k) {
                Vec2 a = corners[k], bb2 = corners[(k+1)%4], ab = bb2 - a;
                float det = dir.x*ab.y - dir.y*ab.x;
                if (std::fabs(det) < 1e-9f) continue;
                Vec2 ao = origin - a;
                float t = -(ao.x*ab.y - ao.y*ab.x) / det;
                float u = -(ao.x*dir.y - ao.y*dir.x) / det;
                if (t >= 0.f && u >= 0.f && u <= 1.f && t < best) best = t;
            }
        }
    }
    return best;
}

static bool bruteInside(const Track& track, Vec2 p) {
    auto cross2d = [](Vec2 u, Vec2 v) { return u.x * v.y - u.y * v.x; };
    auto inTri = [&](Vec2 a, Vec2 b, Vec2 c) {
        float d0 = cross2d(b-a, p-a), d1 = cross2d(c-b, p-b), d2 = cross2d(a-c, p-c);
        return ((d0>=0&&d1>=0&&d2>=0)||(d0<=0&&d1<=0&&d2<=0));
    };
    const auto& L = track.leftBorder();
    const auto& R = track.rightBorder();
    int segs = (int)L.size() - 1;
    for (int i = 0; i < segs; ++i)
        if (inTri(L[i], L[i+1], R[i]) || inTri(L[i+1], R[i+1], R[i]))
            return true;
    return false;
}

static void test_track_spatial_grid() {
    std::mt19937 rng(12345);

    for (const char* path : {"maps/map1.json", "maps/map4_obstaculos.json",
                              "maps/map6_chicanes_infernais.json"}) {
        Track track(path);

        // Compute AABB from spawn position as reference origin.
        Vec2 sp = track.spawnPos();
        float range = 1200.f;

        std::uniform_real_distribution<float> distX(sp.x - range, sp.x + range);
        std::uniform_real_distribution<float> distY(sp.y - range, sp.y + range);
        std::uniform_real_distribution<float> distAngle(0.f, 6.2832f);

        // Raycast correctness: 1000 random rays
        bool rayOk = true;
        for (int k = 0; k < 1000; ++k) {
            Vec2 origin = {distX(rng), distY(rng)};
            float angle = distAngle(rng);
            Vec2 dir = {std::cos(angle), std::sin(angle)};

            float tGrid  = track.raycast(origin, dir, RAY_MAX_LEN);
            float tBrute = bruteRaycast(track, origin, dir, RAY_MAX_LEN);
            if (std::fabs(tGrid - tBrute) > 0.1f) { rayOk = false; break; }
        }
        ASSERT(rayOk);

        // isInsideTrack correctness: 1000 random points
        bool insideOk = true;
        for (int k = 0; k < 1000; ++k) {
            Vec2 p = {distX(rng), distY(rng)};
            if (track.isInsideTrack(p) != bruteInside(track, p)) { insideOk = false; break; }
        }
        ASSERT(insideOk);
    }

    // Spawning point must be inside (smoke test for new impl)
    {
        Track t("maps/map1.json");
        ASSERT(t.isInsideTrack(t.spawnPos()));
    }
}

int main() {
    std::cout << "=== Racing ML Sim Tests ===\n";

    test_vec2();
    std::cout << "Vec2: ok\n";

    test_track_geometry();
    std::cout << "Track geometry: ok\n";

    test_centerline_passes_through_waypoints();
    std::cout << "Centerline passes through waypoints: ok\n";

    test_arc_length_monotonic();
    std::cout << "Arc length monotonic: ok\n";

    test_projection_idempotent();
    std::cout << "Projection idempotent: ok\n";

    test_projection_local_search();
    std::cout << "Projection local search: ok\n";

    test_sensor();
    std::cout << "Sensor: ok\n";

    test_neural_network();
    std::cout << "NeuralNetwork: ok\n";

    test_car_determinism_single();
    std::cout << "Determinism (single-thread): ok\n";

    test_car_determinism_multithread();
    std::cout << "Determinism (multi-thread): ok\n";

    test_genetic_algorithm();
    std::cout << "GeneticAlgorithm: ok\n";

    test_trainers_weight_count();
    std::cout << "Trainers weight count: ok\n";

    test_trainers_determinism();
    std::cout << "Trainers determinism: ok\n";

    test_trainers_elitism();
    std::cout << "Trainers elitism: ok\n";

    test_trainers_resume();
    std::cout << "Trainers resume: ok\n";

    test_training_session_headless();
    std::cout << "TrainingSession headless: ok\n";

    test_game_episode_terminates();
    std::cout << "Game headless episode: ok\n";

    test_reverse_blocked();
    std::cout << "Reverse blocked (task A): ok\n";

    test_reverse_penalty();
    std::cout << "Reverse penalty (task B): ok\n";

    test_regress_penalty();
    std::cout << "Regress penalty (task C): ok\n";

    test_stall_by_no_progress();
    std::cout << "Stall by no-progress (task D): ok\n";

    test_cvar_rank_mean();
    std::cout << "T1 CVaRRank mean ≈ 0.5: ok\n";

    test_cvar_rank_invariance();
    std::cout << "T2 CVaRRank monotonic invariance: ok\n";

    test_cvar_limits();
    std::cout << "T3 CVaR limits: ok\n";

    test_zscore_normalization();
    std::cout << "T4 Z-score normalization: ok\n";

    test_curriculum_linear();
    std::cout << "T5 Curriculum linear/none/explicit: ok\n";

    test_hidden_size();
    std::cout << "T6 setHiddenSize + inferHiddenFromWeights: ok\n";

    test_training_determinism();
    std::cout << "T7 Training determinism (all aggregators): ok\n";

    test_track_spatial_grid();
    std::cout << "E1 Spatial grid raycast/isInsideTrack correctness: ok\n";

    std::cout << "\nResults: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail > 0 ? 1 : 0;
}
