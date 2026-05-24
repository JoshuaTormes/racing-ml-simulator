#ifndef HEADLESS_ONLY
#include "HumanController.h"
#include <SFML/Window/Keyboard.hpp>

Action HumanController::decide(const Observation& /*obs*/) {
    Action a;

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Up)   ||
        sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W))
        a.throttle =  1.f;
    else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Down) ||
             sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S))
        a.throttle = -1.f;

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Left)  ||
        sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A))
        a.steering = -1.f;
    else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Right) ||
             sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D))
        a.steering =  1.f;

    return a;
}

#else
// Stub for headless builds (tests)
#include "HumanController.h"
Action HumanController::decide(const Observation&) { return {}; }
#endif
