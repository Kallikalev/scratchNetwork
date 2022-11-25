#include <vector>
#include <random>

#include "TanhLayer.cpp"

class DenseLayer {
private:
    std::vector<std::vector<float> > weights;
    std::vector<float> biases;
    int numInputs;
    int numNeurons;

    std::vector<std::vector<float> > weightDerivatives;
    std::vector<float> biasDerivatives;


    TanhLayer activation;
public:
    DenseLayer(int _numInputs, int _numNeurons) : activation(_numNeurons) {
        numInputs = _numInputs;
        numNeurons = _numNeurons;

        weights = std::vector<std::vector<float> >(numNeurons);

        biases = std::vector<float>(numNeurons);
        

        // initialize to all defaults
        for (int i = 0; i < numNeurons; i++) {
            weights[i] = std::vector<float>(numInputs);
            for (int j = 0; j < numInputs; j++) {
                weights[i][j] = 1;
            }
        }

        for (int i = 0; i < numNeurons; i++) {
            biases[i] = 0;
        }

        biasDerivatives = std::vector<float>(numNeurons);
        weightDerivatives = std::vector<std::vector<float> >(numNeurons);
        for (int i = 0; i < numNeurons; i++) {
            weightDerivatives[i] = std::vector<float>(numInputs);
        }

    }

    DenseLayer(std::vector<std::vector<float> > _weights, std::vector<float> _biases): activation((int)_weights.size())  {
        numInputs = (int)_weights[0].size();
        numNeurons = (int)_weights.size();
        weights = _weights;
        biases = _biases;

        biasDerivatives = std::vector<float>(numNeurons);
        weightDerivatives = std::vector<std::vector<float> >(numNeurons);
        for (int i = 0; i < numNeurons; i++) {
            weightDerivatives[i] = std::vector<float>(numInputs);
        }
    }

    void randomizeLayer(std::mt19937 gen) {
        std::uniform_real_distribution<> weightDist(-0.3, 0.3);
        std::uniform_real_distribution<> biasDist(-0.3, 0.3);
        for (int i = 0; i < numNeurons; i++) {
            for (int j = 0; j < numInputs; j++) {
                weights[i][j] = (float)weightDist(gen);
            }
            biases[i] = (float)biasDist(gen);
        }
    }

    std::vector<float> forwardPropagate(std::vector<float> input) {
        std::vector<float> output (numNeurons);
        for (int i = 0; i < numNeurons; i++) {
            output[i] = biases[i];
            for (int j = 0; j < numInputs; j++) {
                output[i] += input[j] * weights[i][j];
            }
        }
        return activation.forwardPropagate(output);
    }

    // input is the inputs from the previous layer to the current layer
    // nextDerivatives is derivative of error with respect to each of the neurons in the current layer, size = number of neurons in current layer
    // return value is derivative of error with respect to each of the inputs of the current layer, size = size of input
    std::vector<float> getDerivatives(std::vector<float> input, std::vector<float> output, std::vector<float> nextDerivatives) {
        nextDerivatives = activation.getDerivatives(output,nextDerivatives);
        std::vector<float> inputDerivatives(numInputs);
        
        for (int i = 0; i < numInputs; i++) {
            for (int j = 0; j < numNeurons; j++) {
                inputDerivatives[i] += nextDerivatives[j] * weights[j][i];
            }
        }

        for (int i = 0; i < numNeurons; i++) {
            biasDerivatives[i] += nextDerivatives[i];
        }

        for (int i = 0; i < numNeurons; i++) {
            for (int j = 0; j < numInputs; j++) {
                weightDerivatives[i][j] += input[j] * nextDerivatives[i];
            }
        }

        return inputDerivatives;
    }

    void applyDerivatives(float learnRate) {
        for (int i = 0; i < numNeurons; i++) {
            biases[i] -= biasDerivatives[i] * learnRate;
            biasDerivatives[i] = 0;
            for (int j = 0; j < numInputs; j++) {
                weights[i][j] -= weightDerivatives[i][j] * learnRate;
                weightDerivatives[i][j] = 0;
            }
        }
    }
};