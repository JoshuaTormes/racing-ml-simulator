#pragma once
#include "control/AIController.h"
#include "core/Constants.h"
#include <vector>
#include <string>
#include <cstdint>

inline std::vector<int> defaultTopology() { return {OBS_SIZE, NN_HIDDEN, ACT_SIZE}; }

// Set NN_HIDDEN at startup (before any NeuralNetwork construction). Call at most once.
void setHiddenSize(int h);

// Returns the hidden layer size H that satisfies:
//   OBS_SIZE*H + H + H*ACT_SIZE + ACT_SIZE == weightCount
// Returns -1 if no positive integer H satisfies the equation.
int inferHiddenFromWeights(size_t weightCount);

// Returns the topology stored in a .rnnw file without loading weights.
// Throws std::runtime_error on invalid file or version mismatch.
std::vector<int> readTopologyFromFile(const std::string& path);

// Feedforward MLP, no external libs. Activation: tanh everywhere.
class NeuralNetwork {
public:
    explicit NeuralNetwork(std::vector<int> topology = defaultTopology(),
                           unsigned seed = 0);

    // Forward pass (allocating): input → output (tanh, output ∈ (-1,1))
    std::vector<float> forward(const std::vector<float>& input) const;
    // Forward pass (buffer-reusing): uses output and scratch as ping-pong buffers.
    // Resize is only done when needed; avoids heap allocation on the hot path.
    void forward(const std::vector<float>& input,
                 std::vector<float>& output,
                 std::vector<float>& scratch) const;

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

    Action decide(const Observation& obs) override;
    void   reset() override {}

    NeuralNetwork&       network()       { return nn_; }
    const NeuralNetwork& network() const { return nn_; }

private:
    NeuralNetwork nn_;
    mutable std::vector<float> input_, output_, scratch_;
};
