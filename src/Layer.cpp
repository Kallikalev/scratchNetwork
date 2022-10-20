#include <vector>
#include <random>

class Layer {
private:
    std::vector<std::vector<float> > weights;
    std::vector<float> biases;
    int numInputs;
    int numNeurons;
public:
    std::vector<float> outputs;

    Layer(int _numInputs, int _numNeurons) {
        numInputs = _numInputs;
        numNeurons = _numNeurons;

        weights = std::vector<std::vector<float> >(numNeurons);

        biases = std::vector<float>(numNeurons);
        

        // initialize to all 0s
        for (int i = 0; i < numNeurons; i++) {
            weights[i] = std::vector<float>(numInputs);
            for (int j = 0; j < numInputs; j++) {
                weights[i][j] = 1;
            }
        }

        for (int i = 0; i < numNeurons; i++) {
            biases[i] = 0;
        }

        outputs = std::vector<float>(numNeurons);
    }

    Layer(std::vector<std::vector<float> > _weights, std::vector<float> _biases) {
        numInputs = _weights[0].size();
        numNeurons = _weights.size();
        weights = _weights;
        biases = _biases;
    }

    void randomizeLayer(std::mt19937 gen) {
        std::uniform_real_distribution<> dist(-0.3, 0.3);
        for (int i = 0; i < numNeurons; i++) {
            for (int j = 0; j < numInputs; j++) {
                weights[i][j] = dist(gen);
            }
        }
        
    }

    std::vector<float> runLayer(std::vector<float> inputs) {
        outputs = std::vector<float>(numNeurons);
        for (int i = 0; i < numNeurons; i++) {
            outputs[i] = biases[i];
            for (int j = 0; j < numInputs; j++) {
                outputs[i] += inputs[j] * weights[i][j];
            }
        }
        return outputs;
    }
};