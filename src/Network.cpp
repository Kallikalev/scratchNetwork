#include <vector>

class Network {
public:
    static std::vector<float> runNetwork(std::vector<float> inputs) {

        // initialize trainable properties
        float weight0_1 = 7;
        float weight0_2 = -2.3;
        float bias0 = 2.3;

        float weight1_1 = 0.2;
        float weight1_2 = -0.54;
        float bias1 = 10;

        float oWeight0_1 = 1.0;
        float oWeight0_2 = 1.0;
        float oBias0 = 0;

        float oWeight1_1 = 1.0;
        float oWeight1_2 = 2.0;
        float oBias1 = 0.5;


        // do node operations
        float node0 = inputs[0] * weight0_1 + inputs[1] * weight1_2 + bias0;
        float node1 = inputs[0] * weight1_1 + inputs[1] * weight1_2 + bias1;

        std::vector<float> outputs;


        outputs.push_back(node0 * oWeight0_1 + node1 * oWeight0_2 + oBias0);
        outputs.push_back(node0 * oWeight1_1 + node1 * oWeight1_2 + oBias1);
        return outputs;
    }
};