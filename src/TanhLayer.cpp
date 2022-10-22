#include <vector>
#include <math.h>

class TanhLayer {
private:
    int numNeurons;

    std::vector<float> inputDerivatives;

public:
    std::vector<float> input;
    std::vector<float> output;

    TanhLayer(int _numNeurons) {
        numNeurons = _numNeurons;
    }

    std::vector<float> forwardPropogate(std::vector<float> newInput) {
        input = newInput;
        output = std::vector<float>(numNeurons);
        for (int i = 0; i < numNeurons; i++) {
            output[i] = tanh(input[i]);
        }
        return output;
    }

    std::vector<float> getDerivatives(std::vector<float> nextDerivatives) {
        inputDerivatives = std::vector<float>(numNeurons);
        
        for (int i = 0; i < numNeurons; i++) {
            inputDerivatives[i] = (1 - pow(output[i],2)) * nextDerivatives[i];
        }
        
        return inputDerivatives;
    }
};