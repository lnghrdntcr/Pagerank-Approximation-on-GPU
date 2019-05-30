// Created by Francesco Sgherzi on 15/04/19.
//

#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>

#include <thrust/inner_product.h>

#include "Parse/Parse.h"
#include "Utils/Utils.h"

#define TAU 0.0
#define ALPHA 0.85

#define MAX_B 1024
#define MAX_T 1024

#define MAX_ITER 200

#define num_type double

template<typename T>
bool check_error(T *e, const T error, const unsigned DIMV) {
    for (int i = 0; i < DIMV; ++i) {
        if (e[i] > error) return false;
    }
    return true;
}

template<typename T>
void to_device_csc(T *csc_val, int *csc_non_zero, int *csc_col_idx, const csc_t src) {

    cudaMemcpy(csc_val, &src.val[0], sizeof(T) * src.val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(csc_non_zero, &src.non_zero[0], sizeof(int) * src.non_zero.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(csc_col_idx, &src.col_idx[0], sizeof(int) * src.col_idx.size(), cudaMemcpyHostToDevice);

}

template<typename T>
__global__
void d_set_val(T *m, T value, const unsigned DIMV) {

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (init < DIMV) {

        for (int i = init; i < DIMV; i += stride) {
            m[i] = value;
        }

    }

}

template<typename T>
__global__
void spmv(T *Y, T *pr, T *csc_val, int *csc_non_zero, int *csc_col_idx, const unsigned DIMV) {

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (init < DIMV) {
        for (int i = init; i < DIMV; i += stride) {

            int begin = csc_non_zero[i];
            int end = csc_non_zero[i + 1];

            T acc = 0.0;

            for (int j = begin; j < end; j++) {
                acc += csc_val[j] * pr[csc_col_idx[j]];
            }

            Y[i] = acc;

        }
    }

}


template<typename T>
__global__
void part_spmv(T *Y, T *pr, T *csc_val, int *csc_non_zero, int *csc_col_idx, bool *update_bitmap, const unsigned DIMV) {

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (init < DIMV) {
        for (int i = init; i < DIMV; i += stride) {

            int begin = csc_non_zero[i];
            int end = csc_non_zero[i + 1];
            
            if(update_bitmap[i] == true){

                T acc = 0.0;

                for (int j = begin; j < end; j++) {
                    acc += csc_val[j] * pr[csc_col_idx[j]];
                }

                Y[i] = acc;
            }

        }
    }

}


template<typename T>
__global__
void scale(T *m, T v, const unsigned DIMV) {

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    if (init < DIMV) {
        for (int i = init; i < DIMV; i += stride) {
            m[i] *= v;
        }
    }
}

template<typename T>
__global__
void shift(T *m, T v, const unsigned DIMV) {

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (init < DIMV) {
        for (int i = init; i < DIMV; i += stride) {
            m[i] += v;
        }
    }
}

template<typename T>
__global__
void compute_error(T *error, T *next, T *prev, const unsigned DIMV) {

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (init < DIMV) {
        for (int i = init; i < DIMV; i += stride) {
            error[i] = abs(next[i] - prev[i]);
        }
    }

}

template<typename T>
__global__
void part_compute_error(T *error, T *next, T *prev, bool *update_bitmap, const unsigned DIMV) {

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (init < DIMV) {
        for (int i = init; i < DIMV; i += stride) {
            if(update_bitmap[i]){
                error[i] = abs(next[i] - prev[i]);
                update_bitmap[i] = error[i] > TAU;
            }
        }
    }

}

__global__
void d_set_dangling_bitmap(bool *dangling_bitmap, int *csc_col_idx, const unsigned DIMV) {

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = init; i < DIMV; i += stride) {
        dangling_bitmap[csc_col_idx[i]] = 0;
    }

}


template<typename T1, typename T2>
T2 dot(size_t n, T1 *x, T2 *y) {
    return thrust::inner_product(thrust::device, x, x + n, y, (T2) 0.0);
}

int amain() {


    /**
     * HOST
     */
    num_type *pr;
    num_type *error;

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
    bool     *d_update_bitmap;

    csc_t csc_matrix = parse_dir("/home/fra/University/HPPS/Approximate-PR/graph_generator/generated_csc/smw");

    const unsigned NON_ZERO = csc_matrix.val.size();
    const unsigned DIM = csc_matrix.non_zero.size() - 1;

    std::cout << "\nFEATURES: " << std::endl;
    std::cout << "\tNumber of non zero elements: " << NON_ZERO << std::endl;
    std::cout << "\tNumber of nodes: " << DIM << std::endl;
    std::cout << "\tSparseness: " << (1 - (((double) NON_ZERO) / (DIM * DIM))) * 100 << "%\n" << std::endl;

    cudaMallocHost(&pr, sizeof(num_type) * DIM);
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
    cudaMalloc(&d_update_bitmap, DIM * sizeof(bool));

    std::cout << "Parsing csc files" << std::endl;

    to_device_csc(d_csc_val, d_csc_non_zero, d_csc_col_idx, csc_matrix);
    
    std::cout << "Initializing pr, error, dangling bitmap vectors" << std::endl;

    // Initialize error and pr vector
    cudaMemset(d_pr, (num_type) 1.0 / DIM, DIM);
    cudaMemset(d_error,  (num_type) 1.0, DIM);
    cudaMemset(d_dangling_bitmap, true, DIM);
    cudaMemset(d_update_bitmap, true, DIM);
    
    d_set_dangling_bitmap << < MAX_B, MAX_T >> > (d_dangling_bitmap, d_csc_col_idx, NON_ZERO);


    // Copy them back to their host vectors
    cudaMemcpy(pr, d_pr, DIM * sizeof(num_type), cudaMemcpyDeviceToHost);
    cudaMemcpy(error, d_error, DIM * sizeof(num_type), cudaMemcpyDeviceToHost);

    std::cout << "Beginning pagerank..." << std::endl;

    int iterations = 0;
    bool converged = false;
    while (!converged && iterations < MAX_ITER) {

        // spmv << < MAX_B, MAX_T >> > (d_spmv_res, d_pr, d_csc_val, d_csc_non_zero, d_csc_col_idx, DIM);
        part_spmv <<< MAX_B, MAX_T >>> (d_spmv_res, d_pr, d_csc_val, d_csc_non_zero, d_csc_col_idx, d_update_bitmap, DIM);
        scale << < MAX_B, MAX_T >> > (d_spmv_res, (num_type) ALPHA, DIM);

        // Figure out a way to do the dot product inside GPU
        num_type res_v = dot(DIM, d_dangling_bitmap, d_pr);

        shift << < MAX_B, MAX_T >> > (d_spmv_res, static_cast<num_type> ((1.0 - ALPHA) / DIM + (ALPHA / DIM) * res_v), DIM);

        // compute_error << < MAX_B, MAX_T >> > (d_error, d_spmv_res, d_pr, DIM);
        part_compute_error << < MAX_B, MAX_T >> > (d_error, d_spmv_res, d_pr, d_update_bitmap, DIM);

        cudaDeviceSynchronize();

        cudaMemcpy(error, d_error, DIM * sizeof(num_type), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_pr, d_spmv_res, DIM * sizeof(num_type), cudaMemcpyDeviceToDevice);

        converged = check_error(error, (num_type) TAU, DIM);

        iterations++;
    }

    cudaMemcpy(pr, d_pr, DIM * sizeof(num_type), cudaMemcpyDeviceToHost);


    std::cout << "converged after n_iter: " << iterations << std::endl;


    std::map<int, num_type> pr_map;
    std::vector<std::pair<int, num_type>> sorted_pr;
    std::vector<int> sorted_pr_idxs;

    for (int i = 0; i < DIM; ++i) {
        sorted_pr.push_back({i, pr[i]});
        pr_map[i] = pr[i];
        //std::cout << "Index: " << i << " => " << pr_map[i] << std::endl;
    }

    std::sort(sorted_pr.begin(), sorted_pr.end(),
              [](const std::pair<int, num_type> &l, const std::pair<int, num_type> &r) {
                  if(l.second != r.second)return l.second > r.second;
                  else return l.first > r.first;
              });

    // print the vector
    for (auto const &pair: sorted_pr) {
        sorted_pr_idxs.push_back(pair.first);
    }

    std::cout << "Checking results..." << std::endl;

    std::ifstream results;
    results.open("/home/fra/University/HPPS/Approximate-PR/graph_generator/generated_csc/smw/results.txt");

    int i = 0;
    int tmp = 0;
    int errors = 0;

    while (results >> tmp) {
        // std::cout << "reading " << tmp << std::endl;
        if (tmp != sorted_pr_idxs[i]) {
            errors++;
            // std::cout << "ERROR AT INDEX " << i << ": " << tmp << " != " << sorted_pr_idxs[i] << " Value => " << (num_type) pr_map[sorted_pr_idxs[i]] << std::endl;
        }
        i++;
    }

    std::cout << "Percentage of error: " << (((double) errors) / (DIM)) * 100 << "%\n" << std::endl;

    cudaFree(&pr);
    cudaFree(&error);

    cudaFree(&d_pr);
    cudaFree(&d_error);
    cudaFree(&d_spmv_res);
    cudaFree(&d_csc_val);
    cudaFree(&d_csc_non_zero);
    cudaFree(&d_csc_col_idx);

    return 0;
}
