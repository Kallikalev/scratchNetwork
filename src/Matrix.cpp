//
// Created by Kalev Martinson on 11/26/22.
//
#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <string>
#include <cmath>
template <typename T> class Matrix {
public:
    std::vector<T> data;
    int numRows;
    int numCols;

    // default constructor creates empty Matrix
    Matrix() {
        numRows = 0;
        numCols = 0;
        data = std::vector<T>(0);
    }

    // constructor initializes to all 0s
    explicit Matrix(const std::vector<int> &shape) {
        numRows = shape[0];
        numCols = shape[1];
        data = std::vector<T>(numRows * numCols);
    }

    // initialize matrix to all the same value
    Matrix(const std::vector<int> &shape, T init) {
        numRows = shape[0];
        numCols = shape[1];
        data = std::vector<T>(numRows * numCols);
        for (T & i : data) {
            i = init;
        }
    }

    // copy data from a vector into a matrix
    Matrix(const std::vector<int> &shape, const std::vector<T> &_data) {
        numRows = shape[0];
        numCols = shape[1];
        data = _data;
    }

    // copy the contents of another matrix
    Matrix(const Matrix &m) {
        numRows = m.numRows;
        numCols = m.numCols;
        data = m.data;
    }

    Matrix& set(int row, int col, T val) {
        data[row * numCols + col] = val;
        return *this;
    }

    static Matrix set(const Matrix &m, int row, int col, T val) {
        return Matrix(m).set(row,col,val);
    }

    // sets all data of a matrix to the same value
    Matrix& setAll(T val) {
        for (int i = 0; i < data.size(); i++) {
            data[i] = val;
        }
        return *this;
    }

    static Matrix setAll(const Matrix &m, T val) {
        return Matrix(m).setAll(val);
    }

    // get the value at a specific coordinate of the matrix
    T get(int row, int col) const {
        return data[row * numCols + col];
    }

    static T get(const Matrix &m, int row, int col) {
        return m.get(row,col);
    }

    // extract a row vector from a row of a matrix
    Matrix getRow(int row) const {
        Matrix<T> res ({1,numCols});
        for (int i = 0; i < numCols; i++) {
            res.data[i] = data[row * numCols + i];
        }
        return res;
    }

    static Matrix getRow(const Matrix &m, int row) {
        return m.getRow(row);
    }

    // extract a column vector from a matrix
    Matrix getCol(int col) const {
        Matrix<T> res ({numRows,1});
        for (int i = 0; i < numRows; i++) {
            res.data[i] = data[i * numCols + col];
        }
        return res;
    }

    static Matrix getCol(const Matrix &m, int col) {
        return m.getCol(col);
    }

    // sets a row in a matrix to the data of an equally sized row vector
    Matrix& setRow(int row, const Matrix &rowVector) {
        for (int i = 0; i < numCols; i++) {
            data[row * numCols + i] = rowVector.data[i];
        }
        return *this;
    }

    static Matrix setRow(const Matrix &m, int row, const Matrix &rowVector) {
        return Matrix(m).setRow(row,rowVector);
    }

    // sets a column in a matrix to the data of an equally sized column vector
    Matrix& setCol(int col, const Matrix &colVector) {
        for (int i = 0; i < numRows; i++) {
            set(i,col,colVector.get(i,0));
            data[i * numCols + col] = colVector.data[i];
        }
        return *this;
    }

    static Matrix setCol(const Matrix &m, int col, const Matrix &colVector) {
        return Matrix(m).setCol(col,colVector);
    }

    // perform an inputted function on every element of the matrix
    Matrix& elementOp(T (*func)(T)) {
        for (int i = 0; i < data.size(); i++) {
            data[i] = func(data[i]);
        }
        return *this;
    }

    static Matrix elementOp(const Matrix &m, T (*func)(T)) {
        return Matrix(m).elementOp(func);
    }

    // get the dot product of two matrices by assuming they are 1-dimensional column or row vectors of the same size
    T dot(const Matrix &m2) const {
        T res = 0;
        for (int i = 0; i < data.size(); i++) {
            res += data[i] * m2.data[i];
        }
        return res;
    }

    static T dot(const Matrix &m1, const Matrix &m2) {
        return m1.dot(m2);
    }

    // add a scalar value to every element of the matrix
    Matrix& add(T s) {
        for (T & i : data) {
            i += s;
        }
        return *this;
    }

    static Matrix add(const Matrix &m, T s) {
        return Matrix(m).add(s);
    }

    // subtract a scalar value from every element of the matrix
    Matrix& subtract(T s) {
        return add(-s);
    }

    static Matrix subtract(const Matrix &m, T s) {
        return Matrix::add(m,-s);
    }

    // multiply every element of the matrix by a scalar value
    Matrix& multiply(T s) {
        for (T & i : data) {
            i *= s;
        }
        return *this;
    }

    static Matrix multiply(const Matrix &m, T s) {
        return Matrix(m).multiply(s);
    }

    // divide every element of the matrix by a scalar value
    Matrix& divide(T s) {
        return multiply(1/s);
    }

    static Matrix divide(const Matrix &m, T s) {
        return Matrix::multiply(m, 1/s);
    }

    // exponentiate every element of the matrix by a scalar value
    Matrix& pow(T s) {
        for (T & i : data) {
            i = (T)std::pow(i,s);
        }
        return *this;
    }

    static Matrix pow(const Matrix &m, T s) {
        return Matrix(m).pow(s);
    }

    // add every element of two matrices of the same size
    Matrix& add(const Matrix &m2) {
        for (int i = 0; i < data.size(); i++) {
            data[i] += m2.data[i];
        }
        return *this;
    }

    static Matrix add(const Matrix &m1, const Matrix &m2) {
        return Matrix(m1).add(m2);
    }

    // subtract every element of the second matrix from the first of the same size
    Matrix& subtract(const Matrix &m2) {
        for (int i = 0; i < data.size(); i++) {
            data[i] -= m2.data[i];
        }
        return *this;
    }

    static Matrix subtract(const Matrix &m1, const Matrix &m2) {
        return Matrix(m1).subtract(m2);
    }

    // perform matrix multiplication by getting the dot product of each row and column
    Matrix multiply(const Matrix &m2) const {
        Matrix res ({numRows, m2.numCols});
        for (int row = 0; row < res.numRows; row++) {
            for (int col = 0; col < res.numCols; col++) {
                for (int i = 0; i < numCols; i++) {
                    res.data[row * res.numCols + col] += data[row * numCols + i] * m2.data[i * m2.numCols + col];
                }
            }
        }
        return res;
    }

    static Matrix multiply(const Matrix &m1, const Matrix &m2) {
        return m1.multiply(m2);
    }

    // the Hadamard product, aka element-wise multiplication
    Matrix& hProduct(const Matrix &m2) {
        for (int i = 0; i < data.size(); i++) {
            data[i] *= m2.data[i];
        }
        return *this;
    }

    static Matrix hProduct(const Matrix &m1, const Matrix &m2) {
        return Matrix(m1).hProduct(m2);
    }

    // return a matrix with the same dimensions, rotated 180 degrees. Rotating a matrix is just reversing the underlying vector
    Matrix& rotate180() {
        std::reverse(data.begin(),data.end());
        return *this;
    }

    static Matrix rotate180(const Matrix &m) {
        return Matrix(m).rotate180();
    }

    // transpose a matrix by flipping rows and columns
     Matrix &transpose() {
        *this = transpose(this);
        return *this;
    }

    static Matrix transpose(const Matrix &m) {
        // number of rows and columns are swapped
        Matrix res({m.numCols,m.numRows});
        if (m.numRows == 1 || m.numCols == 1) { // if matrix is a vector, just swap width/height
            res.data = m.data;
            return res;
        }
        for (int i = 0; i < res.numRows; i++) {
            for (int j = 0; j < res.numCols; j++) {
                res.data[i * res.numCols + j] = m.data[j * m.numCols + i];
            }
        }
        return res;
    }

    // return an identity matrix of N * N elements
    static Matrix identity(int n) {
        Matrix res({n,n});
        for (int i = 0; i < n; i++) {
            res.set(n,n,1);
        }
        return res;
    }

    std::string toString() const {
        std::string res;
        for (int row = 0; row < numRows; row++) {
            res += "[";
            for (int col = 0; col < numCols; col++) {
                res += std::to_string(get(row,col));
                if (col != numCols - 1) {
                    res += ",";
                }
            }
            res += "]\n";
        }
        return res;
    }

    static std::string toString(const Matrix &m) {
        return m.toString();
    }
};

#endif