#include <iostream>
// #include <stdio.h>
// #include <math.h>
#include <vector>

#include "Network.cpp"

int main() {
    std::vector<std::vector<int> > networkLayout = {
        {2,3}, // first layer has 2 inputs and 3 neurons
        {3,2}, // second/output layer has 3 inputs and 2 neurons
    };
    Network myNetwork = Network(networkLayout);
    

    // array of inputs and their expected outputs
    std::vector<std::vector<std::vector<float>>> trainingData = {
        {{4,4},{8,0}},
        {{5,4},{9,1}},
        {{2,6},{8,-4}},
        {{6,6},{12,0}},
        {{8,3},{11,5}},
        {{1,1},{2,0}},
        {{5,9},{13,-4}},
        {{3,4},{7,-1}},
        {{4,6},{10,-2}},
        {{2,7},{9,-5}},
        {{1,4},{5,-3}},
        {{5,4},{9,1}},
        {{7,3},{10,4}}
        // {{0,0},{0,1}},
        // {{0,1},{1,0}},
        // {{1,0},{1,0}},
        // {{1,1},{0,0}}
    };


    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            float num1 = (float)i;
            float num2 = (float)j;
            trainingData.push_back({{num1,num2},{num1+num2,num1-num2}});
        }
    }


    myNetwork.randomizeNetwork(); // randomize each time to get good network

    for (int iter = 0; iter < 1000; iter++) {
        float averageLoss = 0;

        for (int n = 0; n < trainingData.size(); n++) {


            std::vector<float> input = trainingData[n][0];
            std::vector<float> expectedOutput = trainingData[n][1];

            float loss = myNetwork.gradientDescent(input,expectedOutput,0.00001);

            averageLoss += loss;

        }

        averageLoss /= trainingData.size();
        
        std::cout << "Average Loss: " << std::to_string(averageLoss) << std::endl;
    }

    std::vector<std::vector<float> > testCases = {
        {0,0},
        {1,0},
        {0,1},
        {2,2},
        {2,4},
        {7,2},
        {0,5},
        {8,3},
        {5,3},
        {4,4}
    };

    for (int i = 0; i < testCases.size(); i++) {
        std::cout << "Input: (" << std::to_string(testCases[i][0]) << ", " << std::to_string(testCases[i][1]) << ")" << std::endl;
        std::vector<float> output = myNetwork.runNetwork(testCases[i]);
        std::cout << "Output: (" << std::to_string(output[0]) << ", " << std::to_string(output[1]) << ")" << std::endl;
    }

    return 0;
}

