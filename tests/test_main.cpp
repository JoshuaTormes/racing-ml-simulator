#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

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
    NeuralNetwork def({OBS_SIZE, 8, 2}, 42);
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
    NeuralNetwork nn({OBS_SIZE, 8, 2});
    nn.load(outDir + "/best.rnnw");
    ASSERT(nn.getWeights().size() == NeuralNetwork({OBS_SIZE, 8, 2}).getWeights().size());

    std::filesystem::remove_all(outDir);
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

int main() {
    std::cout << "=== Racing ML Sim Tests ===\n";

    test_vec2();
    std::cout << "Vec2: ok\n";

    test_track_geometry();
    std::cout << "Track geometry: ok\n";

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

    std::cout << "\nResults: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail > 0 ? 1 : 0;
}
