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

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#define TAU 1e-15
#define ALPHA 0.85

#define MAX_B 1024
#define MAX_T 1024

#define MAX_ITER 200

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

__global__
void d_set_dangling_bitmap(bool *dangling_bitmap, int *csc_non_zero, const unsigned DIMV){

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (init < DIMV){
        for (int i = init; i < DIMV; i += stride) {
            dangling_bitmap[csc_non_zero[i]] = 0;
        }
    }

}

// Tnx parra
template <typename T1, typename T2>
T2 dot(size_t n, T1 *x, T2 *y){
    T2 result = thrust::inner_product(
            thrust::device_pointer_cast(x),
            thrust::device_pointer_cast(x + n),
            thrust::device_pointer_cast(y),
            0.0f);
    return result;
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
    bool     *d_dangling_bitmap;

    csc_t csc_matrix = parse_dir("/home/fra/University/HPPS/Approximate-PR/graph_generator/generated_csc/test");

    const unsigned NON_ZERO = csc_matrix.val.size();
    const unsigned DIM = csc_matrix.non_zero.size() - 1;

    std::cout << "\nFEATURES: " << std::endl;
    std::cout << "\tNumber of non zero elements: " << NON_ZERO << std::endl;
    std::cout << "\tNumber of nodes: " << DIM << std::endl;
    std::cout << "\tSparseness: " << (1 - (((double) NON_ZERO) / (DIM * DIM))) * 100 << "%\n" << std::endl;

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
    cudaMalloc(&d_dangling_bitmap, DIM * sizeof(bool));

    std::cout << "Parsing csc files" << std::endl;

    to_device_csc(d_csc_val, d_csc_non_zero, d_csc_col_idx, csc_matrix);

    std::cout << "Initializing pr, error, dangling bitmap error" << std::endl;

    // Initialize error and pr vector
    d_set_val<<<MAX_B, MAX_T>>>(d_pr, 1.0 / DIM, DIM);
    d_set_val<<<MAX_B, MAX_T>>>(d_error, 1.0, DIM);
    d_set_val<<<MAX_B, MAX_T>>>(d_dangling_bitmap, true, DIM);
    d_set_dangling_bitmap<<<MAX_B, MAX_T>>>(d_dangling_bitmap, d_csc_non_zero, DIM);

    // Copy them back to their host vectors
    cudaMemcpy(pr, d_pr,  DIM * sizeof(num_type), cudaMemcpyDeviceToHost);
    cudaMemcpy(error, d_error, DIM * sizeof(num_type), cudaMemcpyDeviceToHost);


    /**
    * TEST
    */

    std::cout << "Beginning pagerank..." << std::endl;

    int iterations = 0;
    while(!check_error(error, TAU, DIM) && iterations < MAX_ITER){

        // TODO: andare a guardare quali sono i valori ottimali sulla gpu
        spmv<<<MAX_B, MAX_T>>>(d_spmv_res, d_pr, d_csc_val, d_csc_non_zero, d_csc_col_idx, DIM);
        scale<<<MAX_B, MAX_T>>>(d_spmv_res, ALPHA, DIM);

        num_type res_v = dot(DIM, d_dangling_bitmap, d_pr);

        // std::cout << res_v << std::endl;

        shift<<<MAX_B, MAX_T>>>(d_spmv_res, (1.0 - ALPHA) / DIM + (ALPHA / DIM) * /*res_v*/ 0.0, DIM);

        compute_error<<<MAX_B, MAX_T>>>(d_error, d_spmv_res, d_pr, DIM);

        cudaDeviceSynchronize();

        cudaMemcpy(error, d_error, DIM * sizeof(num_type), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_pr, d_spmv_res, DIM * sizeof(num_type), cudaMemcpyDeviceToDevice);

        iterations++;
    }

    cudaMemcpy(pr, d_pr,  DIM * sizeof(num_type), cudaMemcpyDeviceToHost);


    std::cout << "converged after n_iter: " << iterations << std::endl;


    std::map <int, num_type> pr_map;
    std::vector<std::pair<int, num_type>> sorted_pr;
    std::vector<int> sorted_pr_idxs;

    for (int i = 0; i < DIM; ++i) {
        sorted_pr.push_back({i, pr[i]});
        pr_map[i] = pr[i];
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
    results.open("/home/fra/University/HPPS/Approximate-PR/graph_generator/generated_csc/test/results.txt");

    int i = 0;
    int tmp = 0;

    while(results >> tmp){
        if(tmp != sorted_pr_idxs[i]){
            std::cout << "ERROR AT INDEX " << i << ": " << tmp << " != " << sorted_pr_idxs[i] << " Value => " << (double) pr_map[sorted_pr_idxs[i]] << std::endl;
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