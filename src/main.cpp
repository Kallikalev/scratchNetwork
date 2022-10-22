#include <iostream>
// #include <stdio.h>
// #include <math.h>
#include <vector>

#include "Network.cpp"

int main() {
    std::vector<std::vector<int> > networkLayout = {
        {2,3},
        {3,2}
    };
    Network myNetwork = Network(networkLayout);
    

    // array of inputs and their expected outputs
    std::vector<std::vector<std::vector<float>>> trainingData = {
        {{0,0},{0,1}},
        {{0,1},{1,0}},
        {{1,0},{1,0}},
        {{1,1},{0,1}}
    };


    myNetwork.randomizeNetwork(); // randomize each time to get good network

    for (int iter = 0; iter < 100000; iter++) {
        float averageLoss = 0;

        for (int n = 0; n < trainingData.size(); n++) {


            std::vector<float> input = trainingData[n][0];
            std::vector<float> expectedOutput = trainingData[n][1];

            float loss = myNetwork.gradientDescent(input,expectedOutput,0.01);

            averageLoss += loss;

        }

        averageLoss /= trainingData.size();
        
        std::cout << "Average Loss: " << std::to_string(averageLoss) << std::endl;
    }

    std::vector<std::vector<float> > testCases = {
        {0,0},
        {1,0},
        {0,1},
        {1,1}
    };

    for (int i = 0; i < testCases.size(); i++) {
        std::cout << "Input: (" << std::to_string(testCases[i][0]) << ", " << std::to_string(testCases[i][1]) << ")" << std::endl;
        std::vector<float> output = myNetwork.runNetwork(testCases[i]);
        std::cout << "Output: (" << std::to_string(output[0]) << ", " << std::to_string(output[1]) << ")" << std::endl;
    }

    return 0;
}

