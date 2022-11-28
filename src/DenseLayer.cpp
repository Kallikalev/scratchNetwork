#include <random>
#include "Matrix.cpp"

#include "TanhLayer.cpp"

class DenseLayer {
private:
    Matrix<float> weights;
    Matrix<float> biases;
    int numInputs;
    int numNeurons;

    Matrix<float> weightDerivatives;
    Matrix<float> biasDerivatives;


    TanhLayer activation;
public:
    DenseLayer(int _numInputs, int _numNeurons) {
        numInputs = _numInputs;
        numNeurons = _numNeurons;

        weights = Matrix<float>({numNeurons,numInputs},1);
        biases = Matrix<float>({numNeurons,1},0);

        weightDerivatives = Matrix<float>({numNeurons,numInputs},0);
        biasDerivatives = Matrix<float>({numNeurons,1},0);
    }

    DenseLayer(const Matrix<float> &_weights, const Matrix<float> &_biases) {
        numNeurons = _weights.numRows;
        numInputs = _weights.numCols;
        weights = _weights;
        biases = _biases;

        weightDerivatives = Matrix<float>({numNeurons,numInputs},0);
        biasDerivatives = Matrix<float>({numNeurons,1},0);
    }

    void randomizeLayer(std::mt19937 gen) {
        std::uniform_real_distribution<> weightDist(-0.3, 0.3);
        std::uniform_real_distribution<> biasDist(-0.3, 0.3);
        for (int i = 0; i < numNeurons; i++) {
            for (int j = 0; j < numInputs; j++) {
                weights.set(i,j,(float)weightDist(gen));
            }
            biases.set(i,0,(float)biasDist(gen));
        }
    }


    Matrix<float> forwardPropagate(const Matrix<float> &input) {
        Matrix<float> output = weights.multiply(input);
        output.add(biases);
        return TanhLayer::forwardPropagate(output);
    }

    // input is the inputs from the previous layer to the current layer
    // nextDerivatives is derivative of error with respect to each of the neurons in the current layer, size = number of neurons in current layer
    // return value is derivative of error with respect to each of the inputs of the current layer, size = size of input
    Matrix<float> getDerivatives(const Matrix<float> &input, const Matrix<float> &output, const Matrix<float> &nextDerivatives) {
        Matrix<float> activationDerivatives = TanhLayer::getDerivatives(output,nextDerivatives);

        biasDerivatives.add(activationDerivatives);
        weightDerivatives.add(Matrix<float>::multiply(activationDerivatives,Matrix<float>::transpose(input)));

        Matrix<float> inputDerivatives = Matrix<float>::multiply(weights,activationDerivatives);
        return inputDerivatives;
    }

    void applyDerivatives(float learnRate) {
        biases.subtract(Matrix<float>::multiply(biasDerivatives,learnRate));
        biasDerivatives.setAll(0);

        weights.subtract(weightDerivatives.multiply(learnRate));
        weightDerivatives.setAll(0);
    }
};