#pragma once
#ifndef HEADLESS_ONLY

#include "render/Renderer.h"
#include "sim/Game.h"
#include <string>

// Fixed-timestep interactive loop shared by watch/versus/default-windowed modes:
// map-cycling, restart, tick-until-episode-done, render. mapPath selects the
// initial map (for map-cycle index resolution); the game itself is already loaded.
void runInteractiveLoop(Game& game, Renderer& renderer, const std::string& mapPath);

#endif // HEADLESS_ONLY
