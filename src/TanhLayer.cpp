#include <vector>
#include <cmath>

class TanhLayer {
private:
    int numNeurons;

public:
    explicit TanhLayer(int _numNeurons) {
        numNeurons = _numNeurons;
    }

    std::vector<float> forwardPropagate(std::vector<float> input) const {
        std::vector<float> output (numNeurons);
        for (int i = 0; i < numNeurons; i++) {
            output[i] = tanh(input[i]);
        }
        return output;
    }

    std::vector<float> getDerivatives(std::vector<float> output, std::vector<float> nextDerivatives) const {
        std::vector<float> inputDerivatives (numNeurons);
        
        for (int i = 0; i < numNeurons; i++) {
            inputDerivatives[i] = (1.0f - (float)pow(output[i],2)) * nextDerivatives[i];
        }
        
        return inputDerivatives;
    }
};