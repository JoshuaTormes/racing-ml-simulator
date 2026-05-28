#include "NeuralNetwork.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <random>

static constexpr uint32_t NN_VERSION = 2;
static const char         NN_MAGIC[4] = {'R','N','N','W'};

NeuralNetwork::NeuralNetwork(std::vector<int> topology, unsigned seed)
    : topology_(std::move(topology))
{
    initRandom(seed);
}

// Xavier-uniform init: weights ∈ [-limit, limit], biases = 0
void NeuralNetwork::initRandom(unsigned seed) {
    std::mt19937 rng(seed);
    int layers = (int)topology_.size() - 1;
    weights_.resize(layers);
    biases_.resize(layers);
    for (int l = 0; l < layers; ++l) {
        int in  = topology_[l];
        int out = topology_[l + 1];
        float limit = std::sqrt(6.f / (in + out));
        std::uniform_real_distribution<float> dist(-limit, limit);
        weights_[l].resize(out * in);
        for (auto& w : weights_[l]) w = dist(rng);
        biases_[l].assign(out, 0.f);
    }
}

std::vector<float> NeuralNetwork::forward(const std::vector<float>& input) const {
    std::vector<float> cur = input;
    int layers = (int)topology_.size() - 1;
    for (int l = 0; l < layers; ++l) {
        int in  = topology_[l];
        int out = topology_[l + 1];
        std::vector<float> next(out);
        for (int o = 0; o < out; ++o) {
            float sum = biases_[l][o];
            for (int i = 0; i < in; ++i)
                sum += weights_[l][o * in + i] * cur[i];
            next[o] = std::tanh(sum); // tanh activation; output already in (-1,1)
        }
        cur = std::move(next);
    }
    return cur;
}

std::vector<float> NeuralNetwork::getWeights() const {
    std::vector<float> flat;
    int layers = (int)topology_.size() - 1;
    for (int l = 0; l < layers; ++l) {
        flat.insert(flat.end(), weights_[l].begin(), weights_[l].end());
        flat.insert(flat.end(), biases_[l].begin(), biases_[l].end());
    }
    return flat;
}

void NeuralNetwork::setWeights(const std::vector<float>& w) {
    size_t idx = 0;
    int layers = (int)topology_.size() - 1;
    for (int l = 0; l < layers; ++l) {
        size_t wlen = weights_[l].size();
        size_t blen = biases_[l].size();
        if (idx + wlen + blen > w.size())
            throw std::runtime_error("NeuralNetwork::setWeights: vector too short");
        std::copy(w.begin() + idx, w.begin() + idx + wlen, weights_[l].begin());
        idx += wlen;
        std::copy(w.begin() + idx, w.begin() + idx + blen, biases_[l].begin());
        idx += blen;
    }
}

// Serialization: magic(4) + version(4) + nlayers(4) + sizes(4*n) + weights(4*each)
void NeuralNetwork::saveToBuffer(std::vector<uint8_t>& buf) const {
    auto writeBytes = [&](const void* data, size_t n) {
        auto* p = reinterpret_cast<const uint8_t*>(data);
        buf.insert(buf.end(), p, p + n);
    };
    writeBytes(NN_MAGIC, 4);
    uint32_t ver = NN_VERSION;
    writeBytes(&ver, 4);
    uint32_t nl = (uint32_t)topology_.size();
    writeBytes(&nl, 4);
    for (int s : topology_) { uint32_t u = (uint32_t)s; writeBytes(&u, 4); }
    auto flat = getWeights();
    uint32_t nw = (uint32_t)flat.size();
    writeBytes(&nw, 4);
    writeBytes(flat.data(), nw * sizeof(float));
}

void NeuralNetwork::save(const std::string& path) const {
    std::vector<uint8_t> buf;
    saveToBuffer(buf);
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("NeuralNetwork::save: cannot open " + path);
    f.write(reinterpret_cast<const char*>(buf.data()), buf.size());
}

void NeuralNetwork::loadFromBuffer(const std::vector<uint8_t>& buf) {
    size_t pos = 0;
    auto readBytes = [&](void* dst, size_t n) {
        if (pos + n > buf.size()) throw std::runtime_error("NeuralNetwork::load: buffer too short");
        std::memcpy(dst, buf.data() + pos, n);
        pos += n;
    };
    char magic[4];
    readBytes(magic, 4);
    if (std::memcmp(magic, NN_MAGIC, 4) != 0)
        throw std::runtime_error("NeuralNetwork::load: invalid magic");
    uint32_t ver;
    readBytes(&ver, 4);
    if (ver != NN_VERSION)
        throw std::runtime_error("NeuralNetwork::load: unsupported version " + std::to_string(ver));
    uint32_t nl;
    readBytes(&nl, 4);
    std::vector<int> topo(nl);
    for (auto& s : topo) { uint32_t u; readBytes(&u, 4); s = (int)u; }
    if (topo != topology_)
        throw std::runtime_error("NeuralNetwork::load: topology mismatch");
    uint32_t nw;
    readBytes(&nw, 4);
    std::vector<float> flat(nw);
    readBytes(flat.data(), nw * sizeof(float));
    setWeights(flat);
}

void NeuralNetwork::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("NeuralNetwork::load: cannot open " + path);
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), {});
    loadFromBuffer(buf);
}

void setHiddenSize(int h) {
    if (h <= 0) throw std::runtime_error("setHiddenSize: h must be positive, got " + std::to_string(h));
    NN_HIDDEN = h;
}

int inferHiddenFromWeights(size_t weightCount) {
    if (weightCount <= (size_t)ACT_SIZE) return -1;
    size_t num = weightCount - (size_t)ACT_SIZE;
    size_t den = (size_t)(OBS_SIZE + 1 + ACT_SIZE);
    if (num % den != 0) return -1;
    int H = (int)(num / den);
    return H > 0 ? H : -1;
}

std::vector<int> readTopologyFromFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("readTopologyFromFile: cannot open " + path);
    auto readU32 = [&]() -> uint32_t {
        uint32_t v = 0;
        f.read(reinterpret_cast<char*>(&v), 4);
        if (!f) throw std::runtime_error("readTopologyFromFile: unexpected EOF in " + path);
        return v;
    };
    char magic[4];
    f.read(magic, 4);
    if (!f || std::memcmp(magic, NN_MAGIC, 4) != 0)
        throw std::runtime_error("readTopologyFromFile: invalid magic in " + path);
    uint32_t ver = readU32();
    if (ver != NN_VERSION)
        throw std::runtime_error("readTopologyFromFile: unsupported version " + std::to_string(ver) + " in " + path);
    uint32_t nl = readU32();
    std::vector<int> topo(nl);
    for (auto& s : topo) s = (int)readU32();
    return topo;
}

// ---------- NeuralNetworkController ----------

Action NeuralNetworkController::decide(const Observation& obs) {
    std::vector<float> input(obs.begin(), obs.end());
    auto out = nn_.forward(input);
    // Output layer has tanh → already in (-1, 1)
    return Action{
        out.size() > 0 ? out[0] : 0.f,
        out.size() > 1 ? out[1] : 0.f
    };
}
