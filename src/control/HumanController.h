#pragma once
#include "control/AIController.h"

// Reads keyboard input (SFML) and converts to Action. SFML dependency is
// acceptable here since this is an IO adapter, not simulation logic.
class HumanController : public AIController {
public:
    Action decide(const Observation& obs) override;
    void   reset() override {}
};
