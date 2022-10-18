#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>

#include "Network.cpp"

int main() {
    std::array<float,2> inputs = {0.2,-0.3};

    std::array<float,2> outputs = Network::runNetwork(inputs);

    // print to screen
    for (int i = 0; i < outputs.size(); i++) {
        std::cout << std::to_string(outputs[i]) << std::endl;
    }

    return 0;
}

