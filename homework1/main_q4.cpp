#include <iostream>
#include <algorithm>
#include <vector>
#include <list>
#include <numeric>
#include <stdexcept>
#include <cassert>

// Problem 4a
template <typename T>
std::vector<T> daxpy(T a, const std::vector<T>& x, const std::vector<T>& y) {
    std::vector<T> result(x.size());
    if (x.size() != y.size()) {
        throw std::invalid_argument("Sizes don't match.");
    }

    std::transform(x.begin(), x.end(), y.begin(), result.begin(),
        [a](const T &xi, const T& yi) {
            return a  * xi + yi;
        }
    );

    return result;
}

// Problem 4b
constexpr double HOMEWORK_WEIGHT = 0.20;
constexpr double MIDTERM_WEIGHT = 0.35;
constexpr double FINAL_EXAM_WEIGHT = 0.45;

struct Student {
    double homework;
    double midterm;
    double final_exam;
    Student(double hw, double mt, double fe) : homework(hw), midterm(mt), final_exam(fe) {}
};

bool all_students_passed(const std::vector<Student>& students, double pass_threshold) {
    return std::all_of(students.begin(), students.end(), [pass_threshold](const Student &student) {
        double grade = student.homework * HOMEWORK_WEIGHT
                + student.midterm * MIDTERM_WEIGHT
                + student.final_exam * FINAL_EXAM_WEIGHT;
        return grade >= pass_threshold;
    });
}

// Problem 4c
void sort_odd_even(std::vector<int>& data) {
    std::sort(data.begin(), data.end(), [](int lhs, int rhs) {
        bool lhsOdd = lhs % 2 != 0;
        bool rhsOdd = rhs % 2 != 0;
        if (lhsOdd != rhsOdd) return lhsOdd;
        return lhs < rhs;
    });
}

// Problem 4d
template <typename T>
struct SparseMatrixCoordinate {
    int row;
    int col;
    T data;
    SparseMatrixCoordinate(int r, int c, T d) : row(r), col(c), data(d) {}
};

template <typename T>
void sparse_matrix_sort(std::list<SparseMatrixCoordinate<T>>& list) {
    list.sort([](const SparseMatrixCoordinate<T> &a, const SparseMatrixCoordinate<T> &b) {
         if (a.row != b.row) {
             return a.row < b.row;
         }
         return a.col < b.col;
    });
}

int main() {
    // Q4a test
    const int Q4_A = 2;
    const std::vector<int> q4a_x{-2, -1, 0, 1, 2};
    const std::vector<int> q4_y{-2, -1, 0, 1, 2};

    assert((daxpy(Q4_A, q4a_x, q4_y) == std::vector<int>{-6, -3, 0, 3, 6}));
    std::cout << "Q4a test passed!\n";

    // Q4b test
    std::vector<Student> all_pass_students{
        Student(1., 1., 1.),
        Student(0.6, 0.6, 0.6),
        Student(0.8, 0.65, 0.7)
    };

    std::vector<Student> not_all_pass_students{
        Student(1., 1., 1.),
        Student(0, 0, 0)
    };

    assert(all_students_passed(all_pass_students, 0.60));
    assert(!all_students_passed(not_all_pass_students, 0.60));
    std::cout << "Q4b test passed!\n";

    // Q4c test
    std::vector<int> odd_even_sorted = {-5, -3, -1, 1, 3, -4, -2, 0, 2, 4};

    sort_odd_even(odd_even_sorted);
    assert((odd_even_sorted == std::vector<int>{-5, -3, -1, 1, 3, -4, -2, 0, 2, 4}));
    std::cout << "Q4c test passed!\n";

    // Q4d test
    std::list<SparseMatrixCoordinate<int>> sparse{
        SparseMatrixCoordinate<int>(2, 5, 1),
        SparseMatrixCoordinate<int>(2, 2, 2),
        SparseMatrixCoordinate<int>(3, 4, 3)
    };

    sparse_matrix_sort(sparse);
    for (const auto &element : sparse) {
        std::cout << "(" << element.row << ", " << element.col << ", " << element.data << ") ";
    }
    std::cout << "\n";

    std::cout << "Q4d test passed!\n";

    return 0;
}
