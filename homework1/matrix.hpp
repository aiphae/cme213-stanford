#ifndef _MATRIX_HPP
#define _MATRIX_HPP

#include <vector>
#include <stdexcept>
#include <iomanip>
#include <iostream>

template <typename T>
class Matrix {
public:
    virtual ~Matrix() = default;

    virtual T& operator ()(size_t i, size_t j) = 0;
    virtual const T& operator()(size_t i, size_t j) const = 0;

    virtual size_t norm() const = 0;

    virtual size_t size() const = 0;
};

template <typename T>
class MatrixSymmetric : public Matrix<T> {
public:
    MatrixSymmetric(size_t size) : n(size),  data(n * (n + 1) / 2, T()) {}

    T& operator()(size_t i, size_t j) override {
        if (i >= n || j >= n) {
            throw std::out_of_range("Index out of range");
        }
        return data[index(i, j)];
    }

    const T& operator()(size_t i, size_t j) const override {
        if (i >= n || j >= n) {
            throw std::out_of_range("Index out of range");
        }
        return data[index(i, j)];
    }

    size_t norm() const override {
        size_t count = 0;
        for (const auto &val : data) {
            if (val != T()) {
                ++count;
            }
        }
        return count;
    }

    size_t size() const override { return n; }

    friend std::ostream& operator<<(std::ostream &os, const MatrixSymmetric<T> &mat) {
        for (size_t i = 0; i < mat.n; ++i) {
            for (size_t j = 0; j < mat.n; ++j) {
                os << std::setw(6) << mat(i, j) << " ";
            }
            os << "\n";
        }
        return os;
    }

private:
    size_t n;
    std::vector<T> data;

    size_t index(size_t i, size_t j) const {
        if (i > j) std::swap(i, j);
        return j * (j + 1) / 2 + i;
    }
};

#endif /* _MATRIX_HPP */
