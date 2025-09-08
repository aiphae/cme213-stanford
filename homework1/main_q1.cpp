#include "matrix.hpp"
#include <iostream>
#include <cassert>

int main() {
    // Test 1: Construct small matrices
    MatrixSymmetric<double> m1(3);
    assert(m1.size() == 3);
    std::cout << "Test 1 passed: Construction and size.\n";

    // Test 2: Set and get entries, check symmetry
    m1(0, 1) = 5.0;
    assert(m1(1, 0) == 5.0); // Symmetry check
    m1(2, 2) = 10.0;
    assert(m1(2, 2) == 10.0);
    std::cout << "Test 2 passed: Access and symmetry.\n";

    // Test 3: Norm
    assert(m1.norm() == 2); // Only two non-zero stored values
    std::cout << "Test 3 passed: Norm.\n";

    // Test 4: Printing
    std::cout << "Matrix m1:\n" << m1 << "\n";
    std::cout << "Test 4 passed: Printing.";

    // Test 5: Out of bounds
    try {
        m1(3, 0) = 1.0; // Invalid index
        assert(false); // Should not reach here
    }
    catch (const std::out_of_range &) {
        std::cout << "Test 5 passed: Out-of-range exception caught.\n";
    }

    // Test 6: Larger matrix fill
    MatrixSymmetric<int> m2(5);
    for (size_t i = 0; i < m2.size(); ++i) {
        for (size_t j = 0; j <= i; ++j) {
            m2(i, j) = static_cast<int>(i + j);
        }
    }

    std::cout << "Matrix m2:\n" << m2 << "\n";
    std::cout << "m2 norm: " << m2.norm() << "\n";
    assert(m2.norm() > 0);
    std::cout<< "Test 6 passed: Larger matrix fill and norm.\n";

    std::cout << "All test passed successfully!\n";

    return 0;
}