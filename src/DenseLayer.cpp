#include <vector>
#include <random>

#include "TanhLayer.cpp"

class DenseLayer {
private:
    std::vector<std::vector<float> > weights;
    std::vector<float> biases;
    int numInputs;
    int numNeurons;

    std::vector<float> inputDerivatives;
    std::vector<std::vector<float> > weightDerivatives;
    std::vector<float> biasDerivatives;

    TanhLayer activation;
public:
    std::vector<float> input;
    std::vector<float> output;

    DenseLayer(int _numInputs, int _numNeurons) : activation(_numNeurons) {
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

        output = std::vector<float>(numNeurons);
    }

    DenseLayer(std::vector<std::vector<float> > _weights, std::vector<float> _biases): activation(_weights.size())  {
        numInputs = _weights[0].size();
        numNeurons = _weights.size();
        weights = _weights;
        biases = _biases;
    }

    void randomizeLayer(std::mt19937 gen) {
        std::uniform_real_distribution<> weightDist(-0.3, 0.3);
        std::uniform_real_distribution<> biasDist(-0.3, 0.3);
        for (int i = 0; i < numNeurons; i++) {
            for (int j = 0; j < numInputs; j++) {
                weights[i][j] = weightDist(gen);
            }
            biases[i] = biasDist(gen);
        }
    }

    std::vector<float> forwardPropogate(std::vector<float> newInput) {
        input = newInput;
        output = std::vector<float>(numNeurons);
        for (int i = 0; i < numNeurons; i++) {
            output[i] = biases[i];
            for (int j = 0; j < numInputs; j++) {
                output[i] += input[j] * weights[i][j];
            }
        }
        // return activation.forwardPropogate(output);
        return output;
    }

    // input is the inputs from the previous layer to the current layer
    // nextDerivates is derivative of error with respect to each of the neurons in the current layer, size = number of neurons in current layer
    // return value is derivate of error with respect to each of the inputs of the current layer, size = size of input
    std::vector<float> getDerivatives(std::vector<float> nextDerivatives) {
        // nextDerivatives = activation.getDerivatives(nextDerivatives);
        inputDerivatives = std::vector<float>(numInputs);
        
        for (int i = 0; i < numInputs; i++) {
            for (int j = 0; j < numNeurons; j++) {
                inputDerivatives[i] += nextDerivatives[j] * weights[j][i];
            }
        }

        biasDerivatives = std::vector<float>(numNeurons);

        for (int i = 0; i < numNeurons; i++) {
            biasDerivatives[i] = nextDerivatives[i];
        }

        weightDerivatives = std::vector<std::vector<float> >(numNeurons);
        for (int i = 0; i < numNeurons; i++) {
            weightDerivatives[i] = std::vector<float>(numInputs);
            for (int j = 0; j < numInputs; j++) {
                weightDerivatives[i][j] = input[j] * nextDerivatives[i];
            }
        }
        
        return inputDerivatives;
    }

    void applyDerivatives(float learningRate) {
        for (int i = 0; i < numNeurons; i++) {
            biases[i] -= biasDerivatives[i] * learningRate;
            for (int j = 0; j < numInputs; j++) {
                weights[i][j] -= weightDerivatives[i][j] * learningRate;
            }
        }
    }
};