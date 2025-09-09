#include <iostream>
#include <random>
#include <set>

size_t samples_in_range(std::set<double> &set, double lower, double upper) {
    auto lower_bound = set.lower_bound(lower);
    auto upper_bound = set.upper_bound(upper);
    return std::distance(lower_bound, upper_bound);
}

int main() {
    // Test with N(0,1) data
    std::cout << "Generating N(0,1) data" << std::endl;

    std::set<double> data;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (unsigned int i = 0; i < 1000; ++i) {
        data.insert(distribution(generator));
    }

    std::cout << "Number of points in [2, 10]: " << static_cast<int>(samples_in_range(data,  2.0, 10.0)) << "\n";

    return 0;
}
