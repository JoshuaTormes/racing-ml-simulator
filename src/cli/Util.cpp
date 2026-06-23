#include "cli/Util.h"
#include "control/NeuralNetwork.h"
#include <algorithm>
#include <filesystem>
#include <sstream>

std::vector<std::string> listMaps(const std::string& dir) {
    std::vector<std::string> v;
    std::error_code ec;
    for (auto& e : std::filesystem::directory_iterator(dir.empty() ? "maps" : dir, ec))
        if (e.is_regular_file() && e.path().extension() == ".json")
            v.push_back(e.path().string());
    std::sort(v.begin(), v.end());
    return v;
}

int findMapIndex(const std::vector<std::string>& maps, const std::string& cur) {
    auto curName = std::filesystem::path(cur).filename();
    for (size_t i = 0; i < maps.size(); ++i)
        if (std::filesystem::path(maps[i]).filename() == curName) return (int)i;
    return 0;
}

std::vector<std::string> splitComma(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) out.push_back(tok);
    return out;
}

std::string jsonArrayToComma(const nlohmann::json& v) {
    std::string out;
    if (v.is_array()) {
        for (const auto& e : v) { if (!out.empty()) out += ","; out += e.get<std::string>(); }
    } else {
        out = v.get<std::string>();
    }
    return out;
}

std::vector<float> loadChampion(const std::string& path) {
    NeuralNetwork nn(defaultTopology());
    nn.load(path);
    return nn.getWeights();
}
