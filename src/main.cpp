#include <iostream>
// #include <stdio.h>
// #include <math.h>
#include <vector>

#include "Network.cpp"

int main() {
    std::vector<std::vector<int> > networkLayout = {
        {2,3}, // first layer has 2 inputs and 3 neurons
        {3,15}, // second/output layer has 3 inputs and 2 neurons
        {15,30},
        {30,25},
        {25,2}
    };
    Network myNetwork = Network(networkLayout);
    

    // array of inputs and their expected outputs
    std::vector<std::vector<std::vector<float>>> trainingData = {
        {{0,0},{0,0}},
        {{0,1},{1,0}},
        {{1,0},{1,0}},
        {{1,1},{2,0}},
        {{1,2},{3,0}},
        {{2,1},{3,0}}
        // {{0,0},{0,1}},
        // {{0,1},{1,0}},
        // {{1,0},{1,0}},
        // {{1,1},{0,0}}
    };


    myNetwork.randomizeNetwork(); // randomize each time to get good network

    for (int iter = 0; iter < 1000; iter++) {
        float averageLoss = 0;

        for (int n = 0; n < trainingData.size(); n++) {


            std::vector<float> input = trainingData[n][0];
            std::vector<float> expectedOutput = trainingData[n][1];

            float loss = myNetwork.gradientDescent(input,expectedOutput,0.1);

            averageLoss += loss;

        }

        averageLoss /= trainingData.size();
        
        std::cout << "Average Loss: " << std::to_string(averageLoss) << std::endl;
    }

    return 0;
}

