#include <vector>
#include <random>
#include <cmath>

#include "Layer.cpp"

class Network {
private:
    std::vector<Layer> layers;
public:
    /*
    layout is list of layers, each is an ordered pair of (numInputs,numNeurons)
    ex:
    [
        [2,3] two inputs, three neurons, first hidden layer
        [3,5] three inputs, five neurons, second hidden layer
        [5,2] five inputs, two neurons, output layer
    ]
    */

    Network(std::vector<std::vector<int> > layout) {
        for (int i = 0; i < layout.size(); i++) {
            layers.push_back(Layer(layout[i][0],layout[i][1]));
        }
    }

    void randomizeNetwork() {
        
        std::random_device rd;
        std::mt19937 gen(rd());
        for (int i = 0; i < layers.size(); i++) {
            layers[i].randomizeLayer(gen);
        }
    }

    // input size must match input of first layer, output size will be number of nodes of last layer
    std::vector<float> runNetwork(std::vector<float> inputs) {
        // re-use inputs variable
        for (int i = 0; i < layers.size(); i++) {
            inputs = layers[i].runLayer(inputs);
        }
        return inputs;
    }

    static float getLoss(std::vector<float> outputs, std::vector<float> expectedOutputs) {
        float sumError = 0;
        // expectedOutputs and outputs must be the same size
        // mean-squared error formula
        for (int i = 0; i < expectedOutputs.size(); i++) {
            sumError += pow(outputs[i] - expectedOutputs[i],2);
        }

        return sumError / expectedOutputs.size();
    }
};