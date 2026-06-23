#include "app/Runners.h"
#include "cli/AppConfig.h"
#include "cli/ArgParser.h"
#include "cli/Usage.h"
#include <iostream>

int main(int argc, char* argv[]) {
    ParseOutcome pr = parseArgs(argc, argv);
    if (pr.exit) return pr.code;
    if (pr.opt.showHelp) { printUsage(argv[0]); return 0; }

    try {
        AppConfig cfg = resolveConfig(pr.opt);
        return dispatch(cfg);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
