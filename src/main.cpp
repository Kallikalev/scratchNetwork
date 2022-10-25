#include <iostream>
#include <fstream>
// #include <stdio.h>
// #include <math.h>
#include <vector>

#include "Network.cpp"

int main() {
    std::vector<std::vector<int> > networkLayout = {
        {2,3},
        // {3,5},
        // {5,3},
        {3,2}
    };
    Network myNetwork = Network(networkLayout);
    

    // array of inputs and their expected outputs
    std::vector<std::vector<std::vector<float>>> trainingData = {
        {{3,2},{5,1}},
        {{6,2},{8,4}},
        {{1,1},{2,0}},
        {{3,0},{3,3}},
        {{8,8},{16,0}},
        {{3,2},{5,1}},
        {{2,6},{8,-4}},
        {{1,3},{4,-2}},
        {{7,9},{16,-2}},
        {{8,6},{14,2}},
        {{3,5},{8,-2}},
        {{2,7},{9,-5}},
        {{0,3},{3,-3}},
        {{1,1},{2,0}},
        {{7,4},{11,3}}
    };


    myNetwork.randomizeNetwork(); // randomize each time to get good network

    for (int iter = 0; iter < 10000; iter++) {
        float averageLoss = 0;

        for (int n = 0; n < trainingData.size(); n++) {


            std::vector<float> input = trainingData[n][0];
            std::vector<float> expectedOutput = trainingData[n][1];

            float loss = myNetwork.gradientDescent(input,expectedOutput,0.001);

            averageLoss += loss;

        }

        averageLoss /= trainingData.size();
        
        std::cout << "Average Loss: " << std::to_string(averageLoss) << std::endl;
    }

    std::vector<std::vector<float> > testCases = {
        {0,0},
        {0,1},
        {1,0},
        {1,1}
    };

    int precision = 30;
    float max = 10;
    for (int i = 0; i < precision; i++) {
        for (int j = 0; j < precision; j++) {
            testCases.push_back({(float)i/(float)precision * max,(float)j/(float)precision * max});
        }
    }

    // save test cases to CSV file

    std::ofstream outputFile;
    outputFile.open("output.csv");

    // first write number of inputs, then number of outputs
    outputFile << std::to_string(networkLayout[0][0]) << ",";
    outputFile << std::to_string(networkLayout[networkLayout.size()-1][1]);

    for (int i = 0; i < testCases.size(); i++) {
        std::cout << "Input: (" << std::to_string(testCases[i][0]) << ", " << std::to_string(testCases[i][1]) << ")" << std::endl;
        std::vector<float> output = myNetwork.runNetwork(testCases[i]);
        std::cout << "Output: (" << std::to_string(output[0]) << ", " << std::to_string(output[1]) << ")" << std::endl;

        // write each test case to csv file
        for (int j = 0; j < testCases[i].size(); j++) {
            outputFile << "," << std::to_string(testCases[i][j]);
        }
        for (int j = 0; j < output.size(); j++) {
            outputFile << "," << std::to_string(output[j]);
        }      

    }

    outputFile.close();



    return 0;
}

