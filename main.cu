//
// Created by Francesco Sgherzi on 15/04/19.
//

#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <set>
#include <math.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define DIM 1000
#define TAU 0.0
#define ALPHA 0.85

#define num_type double

using namespace std;

// TODO: per dangling nodes => gestiti a parte come contributo costante.
// TODO: vedere se ci sono algoritmi su sklearn per generazione di csc M

template<typename T>
void generate_sparse_matrix(T *matrix, const unsigned int DIMV, const unsigned int min_sparse) {

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


//TODO: Implement generate_sparse_matrix in CUDA
template <typename T>
__global__
void d_generate_sparse_matrix(T *matrix, const unsigned DIMV, const unsigned min_sparse){

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if(init < DIMV){

        for (int i = init; i < DIMV; i += stride) {


            //TODO: here


            // int num_zeroes = rand() % (DIMV - min_sparse) + min_sparse;
            std::set<int> zero_idxs;

        }

    }

}



template <typename T>
void fill_spm(T *matrix, const unsigned int DIMV){
    for (int i = 0; i < DIMV; ++i) {
        int count_zero = 0;
        for (int j = 0; j < DIMV; ++j) {
            if(matrix[i * DIMV + j] == 0.0) count_zero++;
        }
        if(count_zero == DIMV) matrix[i * DIMV + i] = 1;
    }
}

template <typename T>
void transpose(T *out, T *in, const unsigned DIMV){

    for (int i = 0; i < DIMV; ++i) {
        for (int j = 0; j < DIMV; ++j) {
            out[i * DIMV + j] = in[j * DIMV + i];
        }
    }

}

template <typename T>
void to_csc(T *csc_val, int *csc_non_zero, int *csc_col_idx, T* src, const unsigned DIMV, const unsigned non_zero){

    unsigned val_idx = 0;

    csc_non_zero[0] = 0;

    for (int i = 0; i < DIMV; ++i) {

        csc_non_zero[i + 1] = csc_non_zero[i];

        for (int j = 0; j < DIMV; ++j) {

            if(src[i * DIMV + j] > 0){
                csc_val[val_idx] = src[i * DIMV + j];
                csc_non_zero[i + 1]++;
                csc_col_idx[val_idx] = j;

                val_idx++;
            }

        }

    }

    cout << "Bella" << endl;

}

template<typename T>
unsigned int count_non_zero(T *m, const unsigned int DIMV) {
    int sum = 0;

    for (int i = 0; i < DIMV * DIMV; ++i) {
        if (m[i] > 0) sum++;
    }

    return sum;
}

template <typename T>
__global__
void d_set_val( T * m, T value, const unsigned DIMV){

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if(init < DIMV){

        for (int i = init; i < DIMV; i += stride) {
            m[i] = value;
        }

    }

}

template<typename T>
bool check_error(T *e, const T error, const unsigned DIMV) {
    for (int i = 0; i < DIMV; ++i) {
        if(e[i] > error) return false;
    }
    return true;
}

template <typename T>
__global__
void spmv(T *Y, T *pr, T *csc_val, int *csc_non_zero, int *csc_col_idx, const unsigned DIMV){

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if(init < DIMV){
        for(int i = init; i < DIMV; i += stride){

            int begin = csc_non_zero[i];
            int end   = csc_non_zero[i + 1];

            T acc = 0.0;

            for(int j = begin; j < end; j++){
                acc += csc_val[j] * pr[csc_col_idx[j]];
            }

            Y[i] = acc;

        }
    }

}

template <typename T>
__global__
void scale(T *m, T v, const unsigned DIMV){

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if(init < DIMV){
        for (int i = init; i < DIMV; i += stride) {
            m[i] *= v;
        }
    }
}

template <typename T>
__global__
void shift(T *m, T v, const unsigned DIMV){

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if(init < DIMV){
        for (int i = init; i < DIMV; i += stride) {
            m[i] +=v;
        }
    }
}

template <typename T>
__global__
void compute_error(T *error, T *next, T *prev, const unsigned DIMV){

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if(init < DIMV){
        for (int i = init; i < DIMV; i += stride) {
            error[i] = abs(next[i] - prev[i]);
        }
    }

}

int main(){


    /**
     * HOST
     */
    num_type *matrix;
    num_type *matrix_t;
    num_type *pr;
    num_type *spmv_res;
    num_type *error;
    num_type *csc_val;
    int      *csc_non_zero;
    int      *csc_col_idx;
/*
    curandGenerator_t gen;
*/


    /**
     * DEVICE
     */
    num_type *d_pr;
    num_type *d_error;
    num_type *d_spmv_res;
    num_type *d_csc_val;
    int      *d_csc_non_zero;
    int      *d_csc_col_idx;

    cudaMallocHost(&matrix, sizeof(num_type) * DIM * DIM);
    cudaMallocHost(&matrix_t, sizeof(num_type) * DIM * DIM);
    cudaMallocHost(&pr, sizeof(num_type) * DIM);
    cudaMallocHost(&spmv_res, sizeof(num_type) * DIM);
    cudaMallocHost(&error, sizeof(num_type) * DIM);

    // Generate sparse matrix and traspose it
    generate_sparse_matrix(matrix, DIM, DIM * 3 / 4);
    // d_generate_sparse_matrix<<<1, 1>>>(matrix, DIM, DIM - 1);
    transpose(matrix_t, matrix, DIM);
    fill_spm(matrix_t, DIM);

    // Allocate vector for csc matrix
    int non_zero = count_non_zero(matrix_t, DIM);

    cudaMallocHost(&csc_val, sizeof(num_type) * non_zero);
    cudaMallocHost(&csc_non_zero, sizeof(int) * (DIM + 1));
    cudaMallocHost(&csc_col_idx, sizeof(int) * non_zero);

    // Create CSC Matrix
    to_csc(csc_val, csc_non_zero, csc_col_idx, matrix_t, DIM, non_zero);

    // Create device memory
    cudaMalloc(&d_csc_val, sizeof(num_type) * non_zero);
    cudaMalloc(&d_csc_non_zero, sizeof(int) * (DIM + 1));
    cudaMalloc(&d_csc_col_idx, sizeof(num_type) * non_zero);
    cudaMalloc(&d_pr, sizeof(num_type) * DIM);
    cudaMalloc(&d_error, sizeof(num_type) * DIM);
    cudaMalloc(&d_spmv_res, sizeof(num_type) * DIM);

    // Copy from host to device
    cudaMemcpy(d_csc_val, csc_val, sizeof(num_type) * non_zero, cudaMemcpyHostToDevice);
    cudaMemcpy(d_csc_non_zero, csc_non_zero, sizeof(int) * (DIM + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csc_col_idx, csc_col_idx, sizeof(int) * non_zero, cudaMemcpyHostToDevice);

    // Initialize error and pr vector
    d_set_val<<<1, 256>>>(d_pr, 1.0 / DIM, DIM);
    d_set_val<<<1, 256>>>(d_error, 1.0, DIM);

    // Copy them back to their host vectors
    cudaMemcpy(pr, d_pr,  DIM * sizeof(num_type), cudaMemcpyDeviceToHost);
    cudaMemcpy(error, d_error, DIM * sizeof(num_type), cudaMemcpyDeviceToHost);

    int iterations = 0;
    while(!check_error(error, TAU, DIM)){

        // TODO: andare a guardare quali sono i valori ottimali sulla gpu
        spmv<<<256, 256>>>(d_spmv_res, d_pr, d_csc_val, d_csc_non_zero, d_csc_col_idx, DIM);
        scale<<<256, 256>>>(d_spmv_res, ALPHA, DIM);
        shift<<<256, 256>>>(d_spmv_res, (1.0 - ALPHA) / DIM, DIM);

        compute_error<<<256, 256>>>(d_error, d_spmv_res, d_pr, DIM);

        cudaDeviceSynchronize();

        cudaMemcpy(error, d_error, DIM * sizeof(num_type), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_pr, d_spmv_res, DIM * sizeof(num_type), cudaMemcpyDeviceToDevice);
        /*double sum = 0.0;
        for (int i = 0; i < DIM; ++i) {
            //sum += spmv_res[i];
            cout << error[i] << endl;
        }*/
        //cout << sum << endl;

        iterations++;
    }

    cout << "converged after n_iter: " << iterations << endl;

    /*d_set_val<<<1, 256>>>(d_csc_val, 22.0, non_zero);

    cudaMemcpy(csc_val, d_csc_val, sizeof(num_type) * non_zero, cudaMemcpyDeviceToHost);

    cout << "VAL" << endl;
    for (int i = 0; i < non_zero; ++i) {
        cout << csc_val[i] << endl;
    }*/
/*
    cout << "NON_ZERO" << endl;
    for (int i = 0; i < DIM + 1; ++i) {
        cout << csc_non_zero[i] << endl;
    }

    cout << "COL_IDX" << endl;
    for (int i = 0; i < non_zero; ++i) {
        cout << csc_col_idx[i] << endl;
    }

   cout << "NORMAL" << endl;

   for (int i = 0; i < DIM * DIM; ++i) {
       cout << matrix[i] << endl;
   }

   cout << "TRANSPOSEDs" << endl;

   for (int i = 0; i < DIM * DIM; ++i) {
       cout << matrix_t[i] << endl;
   }
*/


    cudaFree(&matrix);
    cudaFree(&matrix_t);
    cudaFree(&pr);
    cudaFree(&spmv_res);
    cudaFree(&error);
    cudaFree(&csc_val);
    cudaFree(&csc_non_zero);
    cudaFree(&csc_col_idx);

    cudaFree(&d_pr);
    cudaFree(&d_error);
    cudaFree(&d_spmv_res);
    cudaFree(&d_csc_val);
    cudaFree(&d_csc_non_zero);
    cudaFree(&d_csc_col_idx);

    return 0;
}