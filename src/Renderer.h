#pragma once
#ifndef HEADLESS_ONLY

#include "Game.h"
#include <SFML/Graphics.hpp>
#include <string>

class Renderer {
public:
    Renderer(unsigned width, unsigned height, const std::string& fontPath);

    // Returns false if window was closed
    bool handleEvents();

    // Draw full scene from game state
    void render(const Game& game, bool showRays = true);

    bool isOpen() const { return window_.isOpen(); }

private:
    sf::RenderWindow window_;
    sf::Font         font_;
    bool             fontLoaded_ = false;

    void drawTrack(const Track& track);
    void drawCar(const Car& car, sf::Color color);
    void drawRays(const Car& car);
    void drawHUD(const Game& game);

    static sf::Vector2f toSf(Vec2 v) { return {v.x, v.y}; }
};

#endif // HEADLESS_ONLY
