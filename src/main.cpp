#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include "Network.cpp"

int main() {

    int numEpochs = 40;
    float learnRate = 0.01;
    int batchSize = 16;

    // read in csv file

    std::vector<std::vector<std::string>> content;
    std::vector<std::string> row;
    std::string line;
    std::string word;

    std::fstream trainingFile("mnist_train.csv", std::ios::in);
    if (trainingFile.is_open()) {
        std::cout << "Reading training data file" << std::endl;
        while (std::getline(trainingFile, line)) {
            row.clear();
            std::stringstream str(line);
            while (std::getline(str,word,',')) {
                row.push_back(word);
            }
            content.push_back(row);
        }
    } else {
        std::cout << "Could not open the file" << std::endl;
        return 0;
    }
    std::cout << "File reading complete" << std::endl;

    std::cout << "Processing training data" << std::endl;

    std::vector<std::vector<std::vector<float>>> trainingData;

    // skip first line of csv file
    for (int i = 1; i < content.size(); i++) {
        row = content[i];

        std::vector<float> pixelInput(784);
        std::vector<float> labelOutput(10);
        labelOutput[std::stoi(row[0])] = 1; // output is 10 numbers, the "correct" one will be 1 and the rest 0. This sets the correct label output to 1
        for (int j = 1; j < 784 + 1; j++) {
            pixelInput[j - 1] = std::stof(row[j])/255; // pixel data is 0-255, convert that to 0-1 value
        }
        trainingData.push_back({pixelInput,labelOutput});
    }

    std::cout << "Training data processing complete" << std::endl;

    std::vector<std::vector<int> > networkLayout = {
//            {784,50},
//            {50,10}
            {784,10}
    };

    Network myNetwork = Network(networkLayout);

    myNetwork.randomizeNetwork(); // randomize each time to get good network

    std::cout << "Running network" << std::endl;

    myNetwork.trainThreaded(trainingData,numEpochs,batchSize,learnRate);

//    std::vector<std::vector<std::string>> content;
//    std::vector<std::string> row;
//    std::string line;
//    std::string word;

    content.clear();

    std::fstream testFile("mnist_test.csv", std::ios::in);
    if (testFile.is_open()) {
        std::cout << "Reading test data file" << std::endl;
        while (std::getline(testFile, line)) {
            row.clear();
            std::stringstream str(line);
            while (std::getline(str,word,',')) {
                row.push_back(word);
            }
            content.push_back(row);
        }
    } else {
        std::cout << "Could not open the file" << std::endl;
        return 0;
    }
    std::cout << "File reading complete" << std::endl;

    std::cout << "Processing test data" << std::endl;

    std::vector<std::vector<std::vector<float>>> testData;

    // skip first line of csv file
    for (int i = 1; i < content.size(); i++) {
        row = content[i];

        std::vector<float> pixelInput(784);
        std::vector<float> labelOutput(10);
        labelOutput[std::stoi(row[0])] = 1; // output is 10 numbers, the "correct" one will be 1 and the rest 0. This sets the correct label output to 1
        for (int j = 1; j < 784 + 1; j++) {
            pixelInput[j - 1] = std::stof(row[j])/255; // pixel data is 0-255, convert that to 0-1 value
        }
        testData.push_back({pixelInput,labelOutput});
    }

    std::cout << "Test data processing complete" << std::endl;

//
//    // save test cases to CSV file
//
//    std::ofstream outputFile;
//    outputFile.open("output.csv");
//
//    // first write number of inputs, then number of outputs
//    outputFile << std::to_string(networkLayout[0][0]) << ",";
//    outputFile << std::to_string(networkLayout[networkLayout.size()-1][1]);

    int numCorrect = 0;

    for (int i = 0; i < testData.size(); i++) {
        std::vector<float> output = myNetwork.runNetwork(testData[i][0]).back();
        int max_input = 0;
        for (int j = 1; j < testData[i][1].size(); j++) {
            if (testData[i][1][j] > testData[i][1][max_input]) {
                max_input = j;
            }
        }
        int max_output = 0;
        for (int j = 1; j < output.size(); j++) {
            if (output[j] > output[max_output]) {
                max_output = j;
            }
        }

        if (max_input == max_output) {
            numCorrect++;
        }

        std::cout << "Predicted: " + std::to_string(max_output) + " Actual: " + std::to_string(max_input) << std::endl;

//        // write each test case to csv file
//        for (int j = 0; j < testCases[i].size(); j++) {
//            outputFile << "," << std::to_string(testCases[i][j]);
//        }
//        for (int j = 0; j < output.size(); j++) {
//            outputFile << "," << std::to_string(output[j]);
//        }

    }

    std::cout << "Accuracy: " + std::to_string((float)numCorrect/(float)testData.size()) << std::endl;

//    outputFile.close();



    return 0;
}

