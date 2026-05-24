#pragma once
#include "core/Types.h"

class AIController {
public:
    virtual ~AIController() = default;
    virtual Action decide(const Observation&) = 0;
    virtual void reset() {}
};
