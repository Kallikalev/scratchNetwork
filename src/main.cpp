#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>


#include "Network.cpp"

int main() {
    std::vector<std::vector<int> > networkLayout(2);
    networkLayout[0] = std::vector<int>(2);
    networkLayout[1] = std::vector<int>(2);
    // first layer has 2 inputs and 3 nodes
    networkLayout[0][0] = 2;
    networkLayout[0][1] = 3;
    // second/output layer has 3 inputs and 2 nodes
    networkLayout[1][0] = 3;
    networkLayout[1][1] = 2;
    Network myNetwork = Network(networkLayout);
    myNetwork.randomizeNetwork();



    for (int y = 0; y < 5; y++) {
        for (int x = 0; x < 5; x++) {
            std::vector<float> inputs(2);
            inputs[0] = x;
            inputs[1] = y;

            std::vector<float> outputs = myNetwork.runNetwork(inputs);

            std::cout << "(" << std::to_string(x) << "," << std::to_string(y) << "): " << std::to_string(outputs[0]) << ", " << std::to_string(outputs[1]) << std::endl;
        }
    }

    return 0;
}

