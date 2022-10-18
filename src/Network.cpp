#include <vector>
#include <array>

class Network {
public:
    static std::array<float,2> runNetwork(std::array<float,2> inputs) {

        // number of neurons, number of inputs
        float weights[2][2] = {
            {7,-2.3}, // neuron 0 input weights
            {0.2,-0.54} // neuron 1 input weights
        };
        // number of neurons
        float biases[2] = {2.3,10};
    
        // number of neurons, number of inputs (which is number of neurons last layer)
        float oWeights[2][2] = {
            {1,1}, // neuron 0 input weights
            {1,2} // neuron 1 input weights
        };
        // number of neurons
        float oBiases[2] = {0,0.5};

        // output of each neuron
        float neuronVals[2];
        for (int i = 0; i < 2; i++) { // number of neurons
            neuronVals[i] = biases[i];
            for (int j = 0; j < 2; j++) { // number of inputs
                neuronVals[0] += inputs[j] * weights[i][j];
            }
        }

        std::array<float,2> outputs;

        for (int i = 0; i < 2; i++) { // number of neurons
            outputs[i] = oBiases[i];
            for (int j = 0; j < 2; j++) { // number of inputs
                outputs[i] += neuronVals[j] * oWeights[i][j];
            }
        }

        return outputs;
    }
};