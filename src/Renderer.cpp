#ifndef HEADLESS_ONLY
#include "Renderer.h"
#include "core/Constants.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <limits>

Renderer::Renderer(unsigned width, unsigned height, const std::string& fontPath)
    : window_(sf::VideoMode({width, height}), "Racing ML Sim")
{
    window_.setFramerateLimit(60);
    fontLoaded_ = font_.openFromFile(fontPath);
}

bool Renderer::handleEvents() {
    while (auto e = window_.pollEvent()) {
        if (e->is<sf::Event::Closed>()) {
            window_.close();
            return false;
        }
        if (const auto* kp = e->getIf<sf::Event::KeyPressed>()) {
            if (kp->code == sf::Keyboard::Key::T)
                toggleTurbo_ = true;
        }
    }
    return window_.isOpen();
}

bool Renderer::consumeToggleTurbo() {
    bool v = toggleTurbo_;
    toggleTurbo_ = false;
    return v;
}

void Renderer::drawTrack(const Track& track) {
    auto drawPoly = [&](const std::vector<Vec2>& pts, sf::Color col) {
        int n = (int)pts.size();
        for (int i = 0; i + 1 < n; ++i) {
            sf::Vertex line[2] = {
                sf::Vertex{toSf(pts[i]),   col},
                sf::Vertex{toSf(pts[i+1]), col}
            };
            window_.draw(line, 2, sf::PrimitiveType::Lines);
        }
    };
    drawPoly(track.leftBorder(),  sf::Color::White);
    drawPoly(track.rightBorder(), sf::Color::White);

    // Waypoints as small dots
    for (auto& wp : track.waypoints()) {
        sf::CircleShape dot(4.f);
        dot.setFillColor(sf::Color(100, 100, 255, 120));
        dot.setPosition({wp.x - 4.f, wp.y - 4.f});
        window_.draw(dot);
    }

    // Obstacles
    for (auto& ob : track.obstacles()) {
        if (ob.type == Obstacle::Type::Circle) {
            sf::CircleShape c(ob.radius);
            c.setFillColor(sf::Color(180, 60, 60, 200));
            c.setPosition({ob.pos.x - ob.radius, ob.pos.y - ob.radius});
            window_.draw(c);
        } else {
            sf::RectangleShape r({ob.size.x, ob.size.y});
            r.setFillColor(sf::Color(180, 60, 60, 200));
            r.setPosition({ob.pos.x - ob.size.x * 0.5f, ob.pos.y - ob.size.y * 0.5f});
            window_.draw(r);
        }
    }
}

void Renderer::drawCar(const Car& car, sf::Color color) {
    sf::RectangleShape rect({20.f, 10.f});
    rect.setFillColor(color);
    rect.setOrigin({10.f, 5.f});
    rect.setPosition(toSf(car.pos));
    rect.setRotation(sf::radians(car.angle));
    window_.draw(rect);
}

void Renderer::drawRays(const Car& car) {
    auto& hits = car.sensor.hitPoints();
    auto& readings = car.sensor.readings();
    for (int i = 0; i < NUM_RAYS; ++i) {
        // Color: green = far, red = near
        float t = readings[i];
        sf::Color col(
            (uint8_t)(255 * (1.f - t)),
            (uint8_t)(255 * t),
            0, 140);
        sf::Vertex line[2] = {
            sf::Vertex{toSf(car.pos),    col},
            sf::Vertex{toSf(hits[i]),    col}
        };
        window_.draw(line, 2, sf::PrimitiveType::Lines);
    }
}

void Renderer::drawHUD(const Game& game) {
    if (!fontLoaded_ || game.cars().empty()) return;
    const Car& c = game.cars()[0];

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);
    ss << "Speed: " << (int)std::fabs(c.speed) << " px/s\n";
    ss << "Time:  " << c.episodeTime << " s\n";
    ss << "Prog:  " << std::setprecision(2) << c.progState.totalProg << "\n";
    ss << "Cars:  " << game.config().population;

    sf::Text text(font_, ss.str(), 16);
    text.setFillColor(sf::Color::White);
    text.setPosition({8.f, 8.f});
    window_.draw(text);
}

void Renderer::render(const Game& game, bool showRays) {
    window_.clear(sf::Color(30, 30, 30));
    drawTrack(game.track());

    const auto& cars = game.cars();
    // In large populations only render up to 200 cars
    int limit = std::min((int)cars.size(), 200);
    for (int i = 0; i < limit; ++i) {
        if (cars[i].done) continue;
        sf::Color col = (i == 0) ? sf::Color::Yellow : sf::Color(100, 200, 100, 180);
        drawCar(cars[i], col);
        if (showRays && i == 0) drawRays(cars[i]);
    }
    drawHUD(game);
    window_.display();
}

void Renderer::renderTraining(const TrainingSession& session, bool turbo) {
    window_.clear(sf::Color(30, 30, 30));
    drawTrack(session.game().track());

    const auto& cars = session.game().cars();
    int limit = std::min((int)cars.size(), 200);
    for (int i = 0; i < limit; ++i) {
        if (cars[i].done) continue;
        sf::Color col = (i == 0) ? sf::Color::Yellow : sf::Color(100, 200, 100, 180);
        drawCar(cars[i], col);
    }

    drawTrainingHUD(session, turbo);
    drawFitnessGraph(session.history());
    window_.display();
}

void Renderer::drawTrainingHUD(const TrainingSession& session, bool turbo) {
    if (!fontLoaded_) return;

    const auto& stats = session.lastStats();
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "Gen: " << session.currentGeneration() + 1 << "/" << session.totalGenerations() << "\n";
    ss << "Algo: " << session.algoName() << "\n";
    if (stats.generation > 0) {
        ss << "Best: " << stats.bestFitness << "\n";
        ss << "Mean: " << stats.meanFitness << "\n";
        ss << "Done: " << stats.completed << "/" << stats.population << "\n";
    }
    ss << (turbo ? "[TURBO]" : "[REALTIME]");

    sf::Text text(font_, ss.str(), 16);
    text.setFillColor(sf::Color::White);
    text.setPosition({8.f, 8.f});
    window_.draw(text);
}

void Renderer::drawFitnessGraph(const std::vector<GenerationStats>& history) {
    if (history.size() < 2) return;

    // Panel in bottom-right: 200x100 px, 10px margin
    auto sz = window_.getSize();
    float pw = 200.f, ph = 100.f, margin = 10.f;
    float px = (float)sz.x - pw - margin;
    float py = (float)sz.y - ph - margin;

    // Background
    sf::RectangleShape bg({pw, ph});
    bg.setPosition({px, py});
    bg.setFillColor(sf::Color(0, 0, 0, 160));
    bg.setOutlineColor(sf::Color(120, 120, 120));
    bg.setOutlineThickness(1.f);
    window_.draw(bg);

    // Find min/max fitness
    float minF =  std::numeric_limits<float>::infinity();
    float maxF = -std::numeric_limits<float>::infinity();
    for (const auto& s : history) {
        if (s.bestFitness < minF) minF = s.bestFitness;
        if (s.bestFitness > maxF) maxF = s.bestFitness;
    }
    float range = (maxF - minF > 1e-6f) ? maxF - minF : 1.f;

    // Draw polyline
    std::vector<sf::Vertex> pts;
    pts.reserve(history.size());
    for (size_t i = 0; i < history.size(); ++i) {
        float t = (float)i / (float)(history.size() - 1);
        float norm = (history[i].bestFitness - minF) / range;
        float x = px + t * pw;
        float y = py + ph - norm * ph;
        pts.push_back(sf::Vertex{{x, y}, sf::Color(100, 220, 100)});
    }
    window_.draw(pts.data(), pts.size(), sf::PrimitiveType::LineStrip);

    // Label
    if (fontLoaded_) {
        sf::Text lbl(font_, "Best Fitness", 11);
        lbl.setFillColor(sf::Color(180, 180, 180));
        lbl.setPosition({px + 4.f, py + 2.f});
        window_.draw(lbl);
    }
}

#endif // HEADLESS_ONLY
