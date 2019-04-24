//
// Created by Francesco Sgherzi on 15/04/19.
//

#include <iostream>
#include <map>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <fstream>
#include <set>
#include <math.h>
#include <algorithm>

#include "Parse/Parse.h"

#include <cuda.h>

#define TAU 10e-6
#define ALPHA 0.85

#define num_type double

using namespace std;

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
template <typename T>
void to_device_csc(T *csc_val, int *csc_non_zero, int *csc_col_idx, const csc_t src){

    cudaMemcpy(csc_val, &src.val[0], sizeof(T) * src.val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(csc_non_zero, &src.non_zero[0], sizeof(int) * src.non_zero.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(csc_col_idx, &src.col_idx[0], sizeof(int) * src.col_idx.size(), cudaMemcpyHostToDevice);

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


    /**
     * DEVICE
     */
    num_type *d_pr;
    num_type *d_error;
    num_type *d_spmv_res;
    num_type *d_csc_val;
    int      *d_csc_non_zero;
    int      *d_csc_col_idx;

    /**
     * TEST
     */
    csc_t csc_matrix = parse_dir("/home/fra/University/HPPS/Approximate-PR/graph_generator/generated_csc/cur");

    const unsigned NON_ZERO = csc_matrix.val.size();
    const unsigned DIM = csc_matrix.non_zero.size() - 1;

    std::cout << "DIMENSIONS: " << std::endl;
    std::cout << "Number of non zero elements: " << NON_ZERO << std::endl;
    std::cout << "Number of nodes: " << DIM << std::endl;
    std::cout << "Sparseness: " << (1 - (((double) NON_ZERO) / (DIM * DIM))) * 100 << "%" << std::endl;

    cudaMallocHost(&matrix, sizeof(num_type) * DIM * DIM);
    cudaMallocHost(&matrix_t, sizeof(num_type) * DIM * DIM);
    cudaMallocHost(&pr, sizeof(num_type) * DIM);
    cudaMallocHost(&spmv_res, sizeof(num_type) * DIM);
    cudaMallocHost(&error, sizeof(num_type) * DIM);


    std::cout << "Initializing device memory" << std::endl;

    // Create device memory
    cudaMalloc(&d_csc_val, sizeof(num_type) * NON_ZERO);
    cudaMalloc(&d_csc_non_zero, sizeof(int) * (DIM + 1));
    cudaMalloc(&d_csc_col_idx, sizeof(num_type) * NON_ZERO);
    cudaMalloc(&d_pr, sizeof(num_type) * DIM);
    cudaMalloc(&d_error, sizeof(num_type) * DIM);
    cudaMalloc(&d_spmv_res, sizeof(num_type) * DIM);

    std::cout << "Parsing csc files" << std::endl;

    to_device_csc(d_csc_val, d_csc_non_zero, d_csc_col_idx, csc_matrix);

    std::cout << "Initializing error vector and pr vector" << std::endl;

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

        iterations++;

        // std::cout << iterations << std::endl;

    }

    cudaMemcpy(pr, d_pr,  DIM * sizeof(num_type), cudaMemcpyDeviceToHost);


    std::cout << "converged after n_iter: " << iterations << std::endl;


    std::map <int, num_type> pr_map;
    std::vector<std::pair<int, num_type>> sorted_pr;
    std::vector<int> sorted_pr_idxs;

    for (int i = 0; i < DIM; ++i) {
        sorted_pr.push_back({i, pr[i]});
        //std::cout << "Index: " << i << " => " << pr_map[i] << std::endl;
    }

    std::sort(sorted_pr.begin(), sorted_pr.end(),
              [](const pair<int, num_type>& l, const pair<int, num_type>& r) {
                  return l.second > r.second;
              });

    // print the vector
    for (auto const &pair: sorted_pr) {
        sorted_pr_idxs.push_back(pair.first);
    }

    std::cout << "Checking results..." << std::endl;

    std::ifstream results;
    results.open("/home/fra/University/HPPS/Approximate-PR/graph_generator/generated_csc/cur/results.txt");

    int i = 0;
    int tmp = 0;

    while(results >> tmp){
        if(tmp != sorted_pr_idxs[i]){
            std::cout << "ERROR " << tmp << " != " << sorted_pr_idxs[i] << std::endl;
        }
        i++;
    }


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