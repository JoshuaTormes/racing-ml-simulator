#include "Game.h"
#include "HumanController.h"
#include "NeuralNetwork.h"
#include "Renderer.h"
#include <iostream>
#include <random>
#include <string>
#include <thread>

static void printUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "  --headless            Run without window\n"
        << "  --map <path>          Map JSON (default: maps/map1.json)\n"
        << "  --population <N>      Number of cars (default: 1)\n"
        << "  --seed <S>            RNG seed (default: 42)\n"
        << "  --benchmark           Run benchmark and exit\n"
        << "  --threads <K>         Thread count (default: hardware_concurrency)\n";
}

int main(int argc, char* argv[]) {
    SimConfig cfg;
    bool benchmark = false;
    bool showHelp  = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--headless")            cfg.headless = true;
        else if (arg == "--benchmark")      { benchmark = true; cfg.headless = true; }
        else if (arg == "--map"      && i+1 < argc) cfg.map        = argv[++i];
        else if (arg == "--population" && i+1 < argc) cfg.population = std::stoi(argv[++i]);
        else if (arg == "--seed"     && i+1 < argc) cfg.seed       = (unsigned)std::stoul(argv[++i]);
        else if (arg == "--threads"  && i+1 < argc) cfg.threads    = std::stoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") showHelp = true;
        else { std::cerr << "Unknown option: " << arg << "\n"; showHelp = true; }
    }

    if (showHelp) { printUsage(argv[0]); return 0; }

    if (benchmark) {
        Game::runBenchmark(cfg);
        return 0;
    }

    if (cfg.headless) {
        Game game(cfg);
        double secs = game.runHeadlessEpisode();
        std::cout << "Headless episode done in " << secs << "s\n";
        return 0;
    }

    // Windowed mode: car 0 controlled by keyboard, rest by NN
    Game game(cfg);
    {
        std::vector<std::unique_ptr<AIController>> ctrls;
        ctrls.push_back(std::make_unique<HumanController>());
        std::mt19937 rng(cfg.seed + 1);
        for (int i = 1; i < cfg.population; ++i)
            ctrls.push_back(std::make_unique<NeuralNetworkController>(
                NeuralNetwork({OBS_SIZE, 8, 2}, (unsigned)rng())));
        game.setControllers(std::move(ctrls));
    }

    Renderer renderer(900, 700, "assets/DejaVuSans.ttf");
    game.reset();

    // Fixed-timestep accumulator: render at display rate, simulate at SIM_HZ
    using Clock = std::chrono::steady_clock;
    auto prev = Clock::now();
    float accumulator = 0.f;

    while (renderer.isOpen()) {
        if (!renderer.handleEvents()) break;

        auto now = Clock::now();
        float dt = std::chrono::duration<float>(now - prev).count();
        prev = now;
        accumulator += dt;

        // Advance simulation in fixed steps
        while (accumulator >= DT) {
            game.tick();
            accumulator -= DT;
            if (game.episodeDone()) {
                game.reset();
                accumulator = 0.f;
                break;
            }
        }

        renderer.render(game, /*showRays=*/true);
    }

    return 0;
}
