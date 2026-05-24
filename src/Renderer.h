#pragma once
#ifndef HEADLESS_ONLY

#include "Game.h"
#include "Training.h"
#include <SFML/Graphics.hpp>
#include <string>

class Renderer {
public:
    Renderer(unsigned width, unsigned height, const std::string& fontPath);

    // Returns false if window was closed; sets toggleTurbo_ if T key pressed
    bool handleEvents();

    // Consume the pending turbo-toggle signal (returns true once, then resets)
    bool consumeToggleTurbo();

    // Draw full scene from game state
    void render(const Game& game, bool showRays = true);

    // Draw training scene (game + HUD + fitness graph)
    void renderTraining(const TrainingSession& session, bool turbo);

    bool isOpen() const { return window_.isOpen(); }

private:
    sf::RenderWindow window_;
    sf::Font         font_;
    bool             fontLoaded_ = false;
    bool             toggleTurbo_ = false;

    void drawTrack(const Track& track);
    void drawCar(const Car& car, sf::Color color);
    void drawRays(const Car& car);
    void drawHUD(const Game& game);
    void drawTrainingHUD(const TrainingSession& session, bool turbo);
    void drawFitnessGraph(const std::vector<GenerationStats>& history);

    static sf::Vector2f toSf(Vec2 v) { return {v.x, v.y}; }
};

#endif // HEADLESS_ONLY
