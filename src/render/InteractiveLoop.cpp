#include "render/InteractiveLoop.h"
#ifndef HEADLESS_ONLY

#include "cli/Util.h"
#include "core/Constants.h"
#include <chrono>
#include <filesystem>

void runInteractiveLoop(Game& game, Renderer& renderer, const std::string& mapPath) {
    game.reset();
    auto mapDir = std::filesystem::path(mapPath).parent_path().string();
    auto maps = listMaps(mapDir);
    int idx = findMapIndex(maps, mapPath);

    using Clock = std::chrono::steady_clock;
    auto prev = Clock::now();
    float accumulator = 0.f;

    while (renderer.isOpen()) {
        if (!renderer.handleEvents()) break;
        if (renderer.consumeRestart()) { game.reset(); accumulator = 0.f; }
        if (int d = renderer.consumeMapDelta(); d && !maps.empty()) {
            idx = (idx + d + (int)maps.size()) % (int)maps.size();
            game.loadMap(maps[idx]); accumulator = 0.f;
        }
        auto now = Clock::now();
        float dt = std::chrono::duration<float>(now - prev).count();
        prev = now;
        accumulator += dt;
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
}

#endif // HEADLESS_ONLY
