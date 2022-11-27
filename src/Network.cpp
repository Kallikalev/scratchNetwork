#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <thread>

#include "DenseLayer.cpp"

class Network {
private:
    std::vector<DenseLayer> layers;
public:
    /*
    layout is list of layers, each is an ordered pair of (numInputs,numNeurons)
    ex:
    [
        [2,3] two inputs, three neurons, first hidden layer
        [3,5] three inputs, five neurons, second hidden layer
        [5,2] five inputs, two neurons, output layer
    ]
    */

    Network(std::vector<std::vector<int> > layout) {
        for (int i = 0; i < layout.size(); i++) {
            layers.push_back(DenseLayer(layout[i][0],layout[i][1]));
        }
    }

    void randomizeNetwork() {
        std::random_device rd;
        std::mt19937 gen(rd());
        for (int i = 0; i < layers.size(); i++) {
            layers[i].randomizeLayer(gen);
        }
    }

    // input size must match input of first layer, output size will be number of neuron of last layer
    std::vector<std::vector<float>> runNetwork(std::vector<float> input) {
        std::vector<std::vector<float>> outputs(layers.size());
        for (int i = 0; i < layers.size(); i++) {
            input = layers[i].forwardPropagate(input);
            outputs[i] = input;
        }
        return outputs;
    }

    static float getLoss(std::vector<float> outputs, std::vector<float> expectedOutputs) {
        float sumError = 0;
        // expectedOutputs and outputs must be the same size
        // mean-squared error formula
        for (int i = 0; i < expectedOutputs.size(); i++) {
            sumError += pow(outputs[i] - expectedOutputs[i],2);
        }

        return sumError / expectedOutputs.size();
    }

    static std::vector<float> getLossGradient(std::vector<float> output, std::vector<float> expectedOutput) {
        std::vector<float> derivatives(output.size());
        for (int i = 0; i < output.size(); i++) {
            derivatives[i] = -2.0f/output.size()*(expectedOutput[i] - output[i]);
        }
        return derivatives;
    }

    float gradientDescent(std::vector<float> input, std::vector<float> expectedOutput) {
        std::vector<std::vector<float>> outputs = runNetwork(input);
        std::vector<float> gradient = getLossGradient(outputs.back(),expectedOutput);

        for (int i = layers.size() - 1; i >= 0; i--) {
            if (i == 0) {
                gradient = layers[i].getDerivatives(input,outputs[0],gradient);
            } else {
                gradient = layers[i].getDerivatives(outputs[i-1],outputs[i],gradient);
            }

        }
        float loss = getLoss(outputs.back(),expectedOutput);
        return loss;
    }

    void gradientDescentThreaded(std::vector<float> input, std::vector<float> expectedOutput, float *averageLoss) {
        std::vector<std::vector<float>> outputs = runNetwork(input);
        std::vector<float> gradient = getLossGradient(outputs.back(),expectedOutput);

        for (int i = layers.size() - 1; i >= 0; i--) {
            if (i == 0) {
                gradient = layers[i].getDerivatives(input,outputs[0],gradient);
            } else {
                gradient = layers[i].getDerivatives(outputs[i-1],outputs[i],gradient);
            }

        }
        float loss = getLoss(outputs.back(),expectedOutput);
        *averageLoss += loss;
    }

    void applyDerivatives(float learnRate) {
        for (int i = 0; i < layers.size(); i++) {
            layers[i].applyDerivatives(learnRate);
        }
    }

    void train(std::vector<std::vector<std::vector<float>>> trainingData, int numEpochs, int batchSize, float learnRate) {
        for (int iter = 0; iter < numEpochs; iter++) {
            float averageLoss = 0;
            // use auto here because I don't want to deal with all the template mess
            auto startTime = std::chrono::high_resolution_clock::now();
            for (int n = 0; n < trainingData.size(); n++) {
                float loss = gradientDescent(trainingData[n][0],trainingData[n][1]);
                averageLoss += loss;

                // apply derivatives in batches, or at the end of the epoch if the epoch size isn't divisible by batch size
                if (n % batchSize == 0 || n == trainingData.size() - 1) {
                    applyDerivatives(learnRate);
                }
            }
            averageLoss /= trainingData.size();

            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);


            std::cout << "Epoch " + std::to_string(iter) + " Complete. Average Loss: " << std::to_string(averageLoss) + ". Time taken: " + std::to_string(duration.count()) + " milliseconds." << std::endl;
        }
    }

    void trainThreaded(std::vector<std::vector<std::vector<float>>> trainingData, int numEpochs, int batchSize, float learnRate) {
        for (int iter = 0; iter < numEpochs; iter++) {
            float averageLoss = 0;
            // use auto here because I don't want to deal with all the template mess
            auto startTime = std::chrono::high_resolution_clock::now();
            for (int n = 0; n < trainingData.size(); n += batchSize) {
                int numThreads = std::min(batchSize,(int)trainingData.size()-n);
                std::vector<std::thread> threads(numThreads);
                for (int i = 0; i < numThreads; i++) {
                    threads[i] = std::thread(&Network::gradientDescentThreaded,this,trainingData[n + i][0],trainingData[n + i][1],&averageLoss);
                }
                for (int i = 0; i < threads.size(); i++) {
                    threads[i].join();
                }
                applyDerivatives(learnRate);
            }
            averageLoss /= trainingData.size();

            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

            std::cout << "Epoch " + std::to_string(iter) + " Complete. Average Loss: " << std::to_string(averageLoss) + ". Time taken: " + std::to_string(duration.count()) + " milliseconds." << std::endl;
        }
    }
};