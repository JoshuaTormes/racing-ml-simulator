#pragma once
#ifndef HEADLESS_ONLY

#include "Game.h"
#include "Training.h"
#include <SFML/Graphics.hpp>
#include <string>

class Renderer {
public:
    Renderer(unsigned width, unsigned height, const std::string& fontPath);

    // Returns false if window was closed; processes the speed-field text input
    bool handleEvents();

    // Current simulation speed multiplier set via the on-screen field (1x default)
    float simSpeed() const { return speedValue_; }

    // Consume pending restart click (returns true once, then resets)
    bool consumeRestart();

    // Consume pending map-cycle delta (-1/0/+1, returns non-zero once, then resets)
    int consumeMapDelta();

    // Draw full scene from game state
    void render(const Game& game, bool showRays = true);

    // Draw training scene (game + HUD + fitness graph)
    void renderTraining(const TrainingSession& session);

    bool isOpen() const { return window_.isOpen(); }

private:
    sf::RenderWindow window_;
    sf::Font         font_;
    bool             fontLoaded_   = false;
    bool             restartClicked_ = false;
    int              mapDelta_      = 0;

    // On-screen "Vel: Nx" text field controlling simulation speed
    sf::FloatRect speedFieldRect_;
    bool          speedFocused_ = false;
    std::string   speedText_;          // digits being typed while focused
    float         speedValue_   = 1.f; // committed multiplier

    sf::FloatRect prevMapBtn_;
    sf::FloatRect restartBtn_;
    sf::FloatRect nextMapBtn_;

    void drawTrack(const Track& track);
    void drawCar(const Car& car, sf::Color color);
    void drawRays(const Car& car);
    void drawHUD(const Game& game);
    void drawTrainingHUD(const TrainingSession& session);
    void drawFitnessGraph(const std::vector<GenerationStats>& history);
    void drawButton(const sf::FloatRect& r, const std::string& label);
    void drawControls(const std::string& mapName);
    void drawSpeedField();
    void commitSpeed();

    static sf::Vector2f toSf(Vec2 v) { return {v.x, v.y}; }
};

#endif // HEADLESS_ONLY
