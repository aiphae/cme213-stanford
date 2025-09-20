#include <stdio.h>

#define N_POINTS 1000
#define MAX_ITER 1000

typedef struct {
    double r;
    double i;
} d_complex;

void test_point(d_complex c);

int num_outside = 0;

int main() {
    double area, error;
    double eps = 1.0e-5;

    // Loop over a grid of points in the complex plane which contains the Mandelbrot set,
    // testing each point to see whether it is inside or outside the set.
    #pragma omp parallel for shared(num_outside)
    for (int i = 0; i < N_POINTS; ++i) {
        for (int j = 0; j < N_POINTS; ++j) {
            d_complex c;
            c.r = -2.0 + 2.5 * (double) (i) / (double) (N_POINTS) + eps;
            c.i = 1.125 * (double) (j) / (double) (N_POINTS) + eps;
            test_point(c);
        }
    }

    // Calculate the area of a set and error estimate and output the results
    area = 2.0 * 2.5 * 1.125 * (double) (N_POINTS * N_POINTS - num_outside) / (double) (N_POINTS * N_POINTS);
    error = area / (double) N_POINTS;

    printf("Area of Mandelbrot set = %12.8f +/- %12.8f\n", area, error);
    printf("Correct answer should be around 1.510659\n");
}

void test_point(d_complex c) {
    // Does the iteration z = z * z + c, until |z| > 2 when the point is known to be outside set
    // If loop count reaches MAX_ITER, the point is considered to be inside the set
    d_complex z = c;
    double temp;

    for (int iter = 0; iter < MAX_ITER; ++iter){
        temp = (z.r * z.r) - (z.i * z.i) + c.r;
        z.i = z.r * z.i * 2 + c.i;
        z.r = temp;
        if ((z.r * z.r + z.i * z.i) > 4.0) {
            #pragma omp atomic
            ++num_outside;
            break;
        }
    }
}