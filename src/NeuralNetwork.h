#pragma once
#include "AIController.h"
#include <vector>
#include <string>
#include <cstdint>

// Feedforward MLP, no external libs. Activation: tanh everywhere.
// Default topology: {10, 8, 2} — matches OBS_SIZE input, 2-float action output.
class NeuralNetwork {
public:
    explicit NeuralNetwork(std::vector<int> topology = {10, 8, 2},
                           unsigned seed = 0);

    // Forward pass: input vector → output vector (tanh, so output ∈ (-1,1))
    std::vector<float> forward(const std::vector<float>& input) const;

    // Flat weight/bias vector access (for GA crossover/mutation)
    std::vector<float> getWeights() const;
    void               setWeights(const std::vector<float>& w);

    const std::vector<int>& topology() const { return topology_; }

    // Versioned binary serialization: magic "RNNW" + uint32 version + topology + float32 weights
    void save(const std::string& path) const;
    void load(const std::string& path);       // throws on incompatible format
    void saveToBuffer(std::vector<uint8_t>& buf) const;
    void loadFromBuffer(const std::vector<uint8_t>& buf);

private:
    std::vector<int>                  topology_; // layer sizes
    // weights_[l] = weight matrix for layer l→l+1, row-major [out][in]
    std::vector<std::vector<float>>   weights_;
    // biases_[l] = bias vector for layer l+1
    std::vector<std::vector<float>>   biases_;

    void initRandom(unsigned seed);
};

// Wraps NeuralNetwork to implement AIController
class NeuralNetworkController : public AIController {
public:
    explicit NeuralNetworkController(NeuralNetwork nn) : nn_(std::move(nn)) {}

    // Convert Observation → NN forward → Action
    Action decide(const Observation& obs) override;
    void   reset() override {}

    NeuralNetwork&       network()       { return nn_; }
    const NeuralNetwork& network() const { return nn_; }

private:
    NeuralNetwork nn_;
};
