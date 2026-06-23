#pragma once
#include "cli/CliOptions.h"

struct ParseOutcome {
    CliOptions opt;
    bool       exit = false;
    int        code = 0;
};

// Auto-detects --config / train.json, applies it, then parses argv (which overrides
// the config file). On config/flag errors sets exit=true with the same exit code and
// std::cerr message main() would have printed before this refactor.
ParseOutcome parseArgs(int argc, char** argv);
