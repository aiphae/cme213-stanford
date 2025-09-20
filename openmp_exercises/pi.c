#include <stdio.h>

double f(double x) {
    return 4.0 / (1 + x * x);
}

long steps = 1000000000;

int main() {
    double lower = 0.0, upper = 1.0;
    double step = (upper - lower) / steps;

    double pi = 0.0;
    #pragma omp parallel for reduction(+: pi)
    for (int i = 0; i < steps; ++i) {
        double x = lower + i * step;
        pi += f(x);
    }

    pi /= steps;

    printf("PI: %.20f\n", pi);
}
