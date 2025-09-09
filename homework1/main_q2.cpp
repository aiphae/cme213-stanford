#include <iostream>
#include <string>
#include <vector>
#include <memory>

class Matrix {
public:
    virtual std::string repr() const = 0;
    virtual ~Matrix() = default;
};

class SparseMatrix : public Matrix {
public:
    std::string repr() const override {
        return "sparse";
    }
};

class ToeplitzMatrix : public Matrix {
public:
    std::string repr() const override {
        return "toeplitz";
    }
};

void PrintRepr(std::vector<std::unique_ptr<Matrix>> &matrices) {
    for (const auto &matrix : matrices) {
        std::cout << matrix->repr() << " ";
    }
    std::cout << "\n";
}

int main() {
    std::vector<std::unique_ptr<Matrix>> matrices;
    matrices.push_back(std::unique_ptr<SparseMatrix>(new SparseMatrix));
    matrices.push_back(std::unique_ptr<ToeplitzMatrix>(new ToeplitzMatrix));

    PrintRepr(matrices);
}
