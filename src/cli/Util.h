#pragma once
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

std::vector<std::string> listMaps(const std::string& dir);
int findMapIndex(const std::vector<std::string>& maps, const std::string& cur);
std::vector<std::string> splitComma(const std::string& s);

// Joins a JSON string-array into a comma-separated string (same format the CLI expects).
std::string jsonArrayToComma(const nlohmann::json& v);

std::vector<float> loadChampion(const std::string& path);
