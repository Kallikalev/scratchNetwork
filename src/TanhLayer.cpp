#include <cmath>
#include "Matrix.cpp"

class TanhLayer {

public:
    static Matrix<float> forwardPropagate(const Matrix<float> &input) {
        return Matrix<float>::elementOp(input,tanh);
    }

    static Matrix<float> getDerivatives(const Matrix<float> &output, const Matrix<float>& nextDerivatives) {

        Matrix<float> inputDerivatives = Matrix<float>::elementOp(output,[](float n) {return 1 - (float)std::pow(n,2.0f);});
        inputDerivatives.hProduct(nextDerivatives);

        return inputDerivatives;
    }
};