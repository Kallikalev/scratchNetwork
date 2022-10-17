#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>

#include "Network.cpp"

int main() {
    std::vector<float> inputs;

    inputs.push_back(0.2);
    inputs.push_back(-0.3);
    std::vector<float> outputs = Network::runNetwork(inputs);


    // print to screen
    std::cout << std::to_string(outputs[0]) << std::endl;
    std::cout << std::to_string(outputs[1]) << std::endl;


    return 0;
}

