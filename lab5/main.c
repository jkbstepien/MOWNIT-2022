//
// Created by jkbstepien on 10.04.2022.
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/times.h>
#include <gsl/gsl_blas.h>

clock_t start_time, end_time;
struct tms timer_start_tms, timer_end_tms;

// Time measurement functions.
void start_timer();
void end_timer();
double calc_time(clock_t start, clock_t end);

// Generate random data for time test.
double** generate_rand_val_square_matrix(int n);
double* generate_rand_val_array(int n);
double* generate_array_with_zeroes(int n);

// Matrix multiplication functions.
double** naive_multiplication(double** A, double** B, int n, int p, int m);
double** better_multiplication(double** A, double** B, int n, int p, int m);

// Free allocated matrix n x m.
void deep_free_matrix(double** A, int n);

int main(int argc, char** argv) {

    FILE* result_file = fopen(argv[1], "w");

    fprintf(result_file, "rep;gsl;better;naive\n");

    for (int n = 100; n <= 500; n += 50) {
        for (int i = 0; i < 10; i++) {

            // Prepare matrices.
            double** A = generate_rand_val_square_matrix(n);
            double** B = generate_rand_val_square_matrix(n);

            // Prepare arrays.
            double* supp_array_c = generate_rand_val_array(n);
            double* supp_array_d = generate_rand_val_array(n);
            double* supp_array_e = generate_array_with_zeroes(n);

            // Prepare matrix view of given array base.
            gsl_matrix_view C = gsl_matrix_view_array(supp_array_c, n, n);
            gsl_matrix_view D = gsl_matrix_view_array(supp_array_d, n, n);
            gsl_matrix_view E = gsl_matrix_view_array(supp_array_e, n, n);

//            start_timer();
            start_time = clock();
            gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &C.matrix, &D.matrix, 0.0, &E.matrix);
//            end_timer();
            end_time = clock();
            fprintf(result_file, "%d;%lf;", n, calc_time(start_time, end_time));
//            start_timer();
            start_time = clock();
            double** F = better_multiplication(A, B, n, n, n);
//            end_timer();
            end_time = clock();
            fprintf(result_file, "%lf;", calc_time(start_time, end_time));
//            start_timer();
            start_time = clock();
            double** G = naive_multiplication(A, B, n, n, n);
//            end_timer();
            end_time = clock();
            fprintf(result_file, "%lf\n", calc_time(start_time, end_time));

            // Free memory.
            free(supp_array_c);
            free(supp_array_d);
            free(supp_array_e);
            deep_free_matrix(A, n);
            deep_free_matrix(B, n);
            deep_free_matrix(F, n);
            deep_free_matrix(G, n);
        }
    }
    fclose(result_file);

    return 0;
}

void start_timer() {
    start_time = clock();
}

void end_timer() {
    end_time = clock();
}

double calc_time(clock_t start, clock_t end) {
    return (double)(end - start)/ CLOCKS_PER_SEC;
}

double** generate_rand_val_square_matrix(int n) {
    srand(time(NULL));
    double** M = (double**) calloc(n, sizeof(double));

    for (int i = 0; i < n; i++) {
        M[i] = (double*) calloc(n, sizeof(double));
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            M[i][j] = (double) rand() / RAND_MAX;
        }
    }

    return M;
}

double* generate_rand_val_array(int n) {
    srand(time(NULL));
    double* M = (double*) calloc(n * n, sizeof(double));

    for (int i = 0; i < n * n; i++) {
        M[i] = (double) rand() / RAND_MAX;
    }

    return M;
}

double* generate_array_with_zeroes(int n) {
    srand(time(NULL));
    double* M = (double*) calloc(n * n, sizeof(double));

    return M;
}

double** naive_multiplication(double** A, double** B, int n, int p, int m) {
    double** C = (double**) calloc(n, sizeof(double));

    for (int i = 0; i < n; i++) {
        C[i] = (double*) calloc(p, sizeof(double));
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < m; k++) {
                C[i][j] = A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

double** better_multiplication(double** A, double** B, int n, int p, int m) {
    double** C = (double**) calloc(n, sizeof(double));

    for (int i = 0; i < n; i++) {
        C[i] = (double*) calloc(p, sizeof(double));
    }

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < m; k++) {
            for (int j = 0; j < p; j++) {
                C[i][j] = C[i][j] + A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

void deep_free_matrix(double** A, int n) {
    for (int i = 0; i < n; i++) {
        free(A[i]);
    }
    free(A);
}

// A – Matrix n x m
// B – Matrix m x p
// C – Matrix n x p