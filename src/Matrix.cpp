//
// Created by Kalev Martinson on 11/26/22.
//
#include <vector>
template <typename T> class Matrix {
public:
    std::vector<T> data;
    int numRows;
    int numCols;

    // empty constructor initializes to all 0s
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

    // get the value at a specific coordinate of the matrix
    static T get(const Matrix &m, int row, int col) {
        return m.data[row * m.numCols + col];
    }

    // extract a row vector from a row of a matrix
    static Matrix getRow(const Matrix &m, int row) {
        std::vector<T> res(m.numCols);
        for (int i = 0; i < m.numCols; i++) {
            res[i] = Matrix::get(m,row,i);
        }
        return Matrix({1,m.numCols},res);
    }

    static Matrix getCol(const Matrix &m, int col) {
        std::vector<T> res(m.numRows);
        for (int i = 0; i < m.numRows; i++) {
            res[i] = Matrix::get(m,i,col);
        }
        return Matrix({m.numRows,1},res);
    }

    // sets a row in a matrix to the data of an equally sized row vector
    static Matrix setRow(const Matrix &m, int row, const Matrix &rowVector) {
        Matrix res(m);
        for (int i = 0; i < m.numCols; i++) {
            res.data[row * m.numCols + i] = rowVector.data[i];
        }
        return res;
    }

    // sets a column in a matrix to the data of an equally sized column vector
    static Matrix setCol(const Matrix &m, int col, const Matrix &colVector) {
        Matrix res(m);
        for (int i = 0; i < m.numRows; i++) {
            res.data[i * m.numCols + col] = colVector.data[i];
        }
        return res;
    }

    // get the dot product of two matrices by assuming they are 1-dimensional column or row vectors of the same size
    static T dot(const Matrix &m1, const Matrix &m2) {
        T res = 0;
        for (int i = 0; i < m1.data.size(); i++) {
            res += m1.data[i] * m2.data[i];
        }
        return res;
    }

    static Matrix set(const Matrix &m, int row, int col, T val) {
        Matrix res(m);
        res.data[row * m.numCols + col] = val;
        return res;
    }

    // add a scalar value to every element of the matrix
    static Matrix add(const Matrix &m, T s) {
        Matrix res(m);
        for (T & i : res.data) {
            i += s;
        }
        return res;
    }

    // subtract a scalar value from every element of the matrix
    static Matrix subtract(const Matrix &m, T s) {
        return Matrix::add(m,-s);
    }

    // multiply every element of the matrix by a scalar value
    static Matrix multiply(const Matrix &m, T s) {
        Matrix res(m);
        for (T & i : res.data) {
            i *= s;
        }
        return res;
    }

    // divide every element of the matrix by a scalar value
    static Matrix divide(const Matrix &m, T s) {
        return Matrix::multiply(m, 1/s);
    }

    // add every element of two matrices of the same size
    static Matrix add(const Matrix &m1, const Matrix &m2) {
        Matrix res(m1);
        for (int i = 0; i < m1.data.size(); i++) {
            res.data[i] += m2.data[i];
        }
        return res;
    }

    // subtract every element of the second matrix from the first of the same size
    static Matrix subtract(const Matrix &m1, const Matrix &m2) {
        return Matrix::add(m1,Matrix::multiply(m2,-1));
    }

    // perform matrix multiplication by getting the dot product of each row and column
    static Matrix multiply(const Matrix &m1, const Matrix &m2) {
        Matrix res ({m1.numRows, m2.numCols});
        for (int row = 0; row < res.numRows; row++) {
            for (int col = 0; col < res.numCols; col++) {
                res = set(res,row,col,dot(getRow(m1,row),getCol(m2,col)));
            }
        }
        return res;
    }

    // return a matrix with the same dimensions, rotated 180 degrees. Rotating a matrix is just reversing the underlying vector
    static Matrix rotate180(const Matrix &m) {
        Matrix res(m);
        for (int i = 0; i < m.data.size(); i++){
            res.data[i] = m.data[m.data.size() - i - 1];
        }
        return res;
    }

    // transpose a matrix by flipping rows and columns
    static Matrix transpose(const Matrix &m) {
        // number of rows and columns are swapped
        Matrix res({m.numCols,m.numRows});
        for (int i = 0; i < res.numCols; i++) {
            res = setCol(res,i,Matrix::getRow(m,i));
        }
        return res;
    }

    // return an identity matrix of N * N elements
    static Matrix identity(int n) {
        Matrix res({n,n});
        for (int i = 0; i < n; i++) {
            res = Matrix::set(res,n,n,1.0f);
        }
        return res;
    }

    static std::string toString(Matrix &m) {
        std::string res;
        for (int row = 0; row < m.numRows; row++) {
            res += "[";
            for (int col = 0; col < m.numCols; col++) {
                res += std::to_string(Matrix::get(m,row,col));
                if (col != m.numCols - 1) {
                    res += ",";
                }
            }
            res += "]\n";
        }
        return res;
    }
};