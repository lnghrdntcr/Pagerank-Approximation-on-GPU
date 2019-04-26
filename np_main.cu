#include <iostream>

#include <stdlib.h>
#include <time.h>
#include <set>
#include <math.h>


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#define DIM 10
#define TAU 1e-10
#define ALPHA 0.85
#define num_type double

template<typename T>
void generate_sparse_matrix(thrust::host_vector <T> &matrix, const unsigned int DIMV, const unsigned int min_sparse) {

    // for all rows
    for (int i = 0; i < DIMV; ++i) {

        int num_zeroes = rand() % (DIMV - min_sparse) + min_sparse;
        std::set<int> zero_idxs;

        zero_idxs.insert(i);
        for (int j = 0; j < num_zeroes; ++j) {
            int r_idx = rand() % DIMV;
            zero_idxs.insert(r_idx);
        }

        for (int j = 0; j < DIMV; ++j) {
            if (zero_idxs.find(j) == zero_idxs.end() && (DIMV - zero_idxs.size()) != 0) {
                matrix[i * DIMV + j] = (T) 1.0 / (DIMV - zero_idxs.size());
            }
        }

    }
}

template <typename T>
void fill_spm(thrust::host_vector <T> &matrix, const unsigned int DIMV){
    for (int i = 0; i < DIMV; ++i) {
        int count_zero = 0;
        for (int j = 0; j < DIMV; ++j) {
            if(matrix[i * DIMV + j] == 0.0) count_zero++;
        }
        if(count_zero == DIMV) matrix[i * DIMV + i] = 1;
    }
}

template<typename T>
void transpose(thrust::host_vector <T> &out, thrust::host_vector <T> in, const unsigned int DIMV) {

    for (int i = 0; i < DIMV; ++i) {

        for (int j = 0; j < DIMV; ++j) {
            out[i * DIMV + j] = in[j * DIMV + i];
        }

    }

}

template<typename T>
unsigned int num_non_zero(thrust::host_vector <T> m, const unsigned int DIMV) {
    int sum = 0;

    for (int i = 0; i < DIMV * DIMV; ++i) {
        if (m[i] > 0) sum++;
    }

    return sum;
}

template<typename T>
void to_csc(thrust::host_vector <T> &csc, thrust::host_vector <T> matrix, const unsigned int non_zero,
            const unsigned int DIMV) {

    unsigned val_idx = 0;
    unsigned non_zero_idx = non_zero;
    unsigned col_idx = non_zero + DIMV + 1;

    csc[non_zero_idx] = 0;

    for (int i = 0; i < DIMV; ++i) {

        csc[non_zero_idx + 1] = csc[non_zero_idx];
        non_zero_idx++;

        for (int j = 0; j < DIMV; ++j) {

            if (matrix[i * DIMV + j] > 0) {
                csc[val_idx] = matrix[i * DIMV + j];
                csc[non_zero_idx]++;
                csc[col_idx] = j;

                val_idx++;
                col_idx++;

            }

        }

    }

}

template<typename T>
bool check_error(thrust::host_vector <T> e, const T error) {
    for (auto el: e) {
        if (el > error) return false;
    }
    return true;
}

template<typename T>
T *to_raw(thrust::device_vector <T> v) {
    return thrust::raw_pointer_cast(&v[0]);
}

/*
template<typename T>
__global__
void spmv(T * Y, T *csc, T *pr, const unsigned int DIMV, const unsigned non_zero) {

    unsigned val_idx = 0;
    unsigned non_zero_idx = non_zero;
    unsigned col_idx = non_zero + DIMV + 1;

    for (int i = 0; i < DIMV; ++i) {

        int begin = (int) csc[non_zero_idx + i];
        int end = (int) csc[non_zero_idx + i + 1];

        T acc = 0.0;
        for (int j = begin; j < end; ++j) {
            acc += csc[val_idx + j] * pr[(int) csc[col_idx + j]];
        }

        (Y[i]) = acc;

    }

}
*/

template<typename T>
void pagerank(thrust::host_vector <T> &Y, thrust::host_vector <T> &error, thrust::host_vector <T> csc,
              thrust::host_vector <T> pr, const unsigned DIMV, const unsigned non_zero) {

    unsigned val_idx = 0;
    unsigned non_zero_idx = non_zero;
    unsigned col_idx = non_zero + DIMV + 1;

    for (int i = 0; i < DIMV; ++i) {

        int begin = (int) csc[non_zero_idx + i];
        int end = (int) csc[non_zero_idx + i + 1];

        T acc = 0.0;
        for (int j = begin; j < end; ++j) {
            acc += csc[val_idx + j] * pr[(int) csc[col_idx + j]];
        }

        Y[i] = (1 - ALPHA) / DIM + ALPHA * acc;
        error[i] = abs(pr[i] - Y[i]);
    }

}

int np_main() {

    srand(time(NULL));

    /**
     * HOST SETUP
     */
    thrust::host_vector<num_type> matrix(DIM * DIM, 0);
    thrust::host_vector<num_type> matrix_t(DIM * DIM, 0);
    thrust::host_vector<num_type> pr(DIM, 1.0 / DIM);
    thrust::host_vector<num_type> error(DIM, 1.0);
    thrust::host_vector<num_type> Y(DIM, 1.0 / DIM);


    generate_sparse_matrix(matrix, DIM, DIM - 1);

    /*std::cout << "MATRIX" << std::endl;
    for (int j = 0; j < DIM * DIM; ++j) {
        std::cout << matrix[j]  << std::endl;
    }

    fill_spm(matrix, DIM);

    std::cout << "MATRIX" << std::endl;
    for (int j = 0; j < DIM * DIM; ++j) {
        std::cout << matrix[j]  << std::endl;
    }*/

    transpose(matrix_t, matrix, DIM);

    const unsigned int non_zero = num_non_zero(matrix_t, DIM);

    thrust::host_vector<num_type> csc(non_zero * 2 + DIM + 1);

    to_csc(csc, matrix_t, non_zero, DIM);

    /**
     * DEVICE SETUP
     * A cudaMemcpy is happening where the `=` sign is.
     */

    int iterations = 0;
    while (!check_error(error, TAU)) {

        thrust::host_vector<num_type> Y(DIM, 1.0 / DIM);

        pagerank(Y, error, csc, pr, DIM, non_zero);

        thrust::copy(Y.begin(), Y.end(), pr.begin());

        iterations++;

    }

/*
    for (int i = 0; i < DIM; ++i) {
        std::cout << i << ": " << pr[i] << std::endl;
    }
*/

    std::cout << "Converged after n_iter: " << iterations << std::endl;
/*
    for (auto el: matrix) {
        std::cout << el << std::endl;
    }

    std::cout << "TRANSPOSED" << std::endl;

    for (auto el: matrix_t) {
        std::cout << el << std::endl;
    }

    std::cout << "CSC" << std::endl;

    for (auto el: csc) {
        std::cout << el << std::endl;
    }*/
    
    return 0;
}