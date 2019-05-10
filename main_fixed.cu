// Created by Francesco Sgherzi on 15/04/19.
//

// TODO: Check graphblast
// TODO: Implement spmv with bitmask graphblast
// TODO: Different data types.

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

#define MAX_B 1
#define MAX_T 1

#define MAX_ITER 200

#define num_type unsigned

// 00.00 0000 0000 0000 0000 0000 0000 0000
#define SCALE 30

// #define d_to_fixed(x) ((x) * ( (double) (1 << SCALE)))
#define fixed_mult(x, y) ((((x) >> SCALE / 2) * ((y) >> SCALE / 2)) >> 0)

// TODO: Figure out a way to make fixed point division work
// #define fixed_div(x, y) ((x << SCALE / 2) / (y << SCALE / 2))

__host__
__device__
inline unsigned d_to_fixed(double x) {

    return x * ((double) (1 << SCALE));

}

csc_fixed_t to_fixed_csc(csc_t m) {

    csc_fixed_t fixed_csc;

    fixed_csc.col_idx = m.col_idx;
    fixed_csc.non_zero = m.non_zero;
    fixed_csc.val = std::vector<unsigned>();

    for (int i = 0; i < m.val.size(); ++i) {
        fixed_csc.val.push_back(d_to_fixed(m.val[i]));
    }

    return fixed_csc;

}

template<typename T>
void to_device_csc(T *csc_val, int *csc_non_zero, int *csc_col_idx, const csc_fixed_t src) {

    cudaMemcpy(csc_val, &src.val[0], sizeof(T) * src.val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(csc_non_zero, &src.non_zero[0], sizeof(int) * src.non_zero.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(csc_col_idx, &src.col_idx[0], sizeof(int) * src.col_idx.size(), cudaMemcpyHostToDevice);

}

__global__
void d_fixed_set_dangling_bitmap(bool *dangling_bitmap, int *csc_col_idx, const unsigned DIMV) {

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = init; i < DIMV; i += stride) {
        dangling_bitmap[csc_col_idx[i]] = 0;
    }

}


template<typename T>
__global__
void d_fixed_spmv(T *Y, T *pr, T *csc_val, int *csc_non_zero, int *csc_col_idx, const int DIMV) {

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;


    for (int i = init; i < DIMV; i += stride) {

        int begin = csc_non_zero[i];
        int end = csc_non_zero[i + 1];

        T acc = d_to_fixed(0.0);

        for (int j = begin; j < end; ++j) {
            acc += fixed_mult(csc_val[j], pr[csc_col_idx[j]]);
        }
        Y[i] = acc;
    }
}

// Until I figure out how memset works
template<typename T>
__global__
void d_set_value(T *v, const T value, const unsigned DIMV) {

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;


    for (int i = init; i < DIMV; i += stride) {
        v[i] = value;
    }

}

template<typename T>
__global__
void d_fixed_scale(T *v, T value, const unsigned DIMV) {

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = init; i < DIMV; i += stride) {
        v[i] = fixed_mult(v[i], value);
    }

}

template<typename T>
__global__
void d_fixed_shift(T *v, T value, const unsigned DIMV) {

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = init; i < DIMV; i += stride) {
        v[i] = v[i] + value;
    }

}

__device__
        __forceinline__

unsigned d_fixed_abs(const unsigned x, const unsigned y) {
    if (x > y) return x - y;
    else return y - x;
}

template<typename T>
__global__
void d_fixed_compute_error(T *error, T *v1, T *v2, const unsigned DIMV) {

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = init; i < DIMV; i += stride) {
        error[i] = d_fixed_abs(v1[i], v2[i]);
    }

}

template<typename T>
bool check_error(T *e, const T error, const unsigned DIMV) {
    for (int i = 0; i < DIMV; ++i) {
        // printf("%d => %d\n", e[i], error);
        if (e[i] > error) return false;
    }
    return true;
}

template <typename T1, typename T2>
T1 h_fixed_dot(T1 *v1, T2 *v2, const unsigned DIMV){

    T1 *h_v1;
    T2 *h_v2;

    T1 res = d_to_fixed(0.0);

    cudaMallocHost(&h_v1, sizeof(T1) * DIMV);
    cudaMallocHost(&h_v2, sizeof(T2) * DIMV);

    for (int i = 0; i < DIMV; ++i) {
        res += fixed_mult(h_v1[i], h_v2[i]);
    }

    cudaFree(&h_v1);
    cudaFree(&h_v2);

    return res;

}

template<typename T>
void debug_print(char *name, T *v, const unsigned DIMV) {

    T *test;
    cudaMallocHost(&test, DIMV * sizeof(num_type));
    cudaMemcpy(test, v, DIMV * sizeof(num_type), cudaMemcpyDeviceToHost);

    std::cout << "---------------------DEBUG:" << name << "-------------------" << std::endl;
    for (int i = 0; i < DIMV; ++i) {

        std::cout << test[i] << std::endl;

    }
    std::cout << "------------------END DEBUG:" << name << "-------------------" << std::endl;

}

int main() {



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
    int *d_csc_non_zero;
    int *d_csc_col_idx;
    bool *d_dangling_bitmap;
    bool *d_update_bitmap;

    csc_t csc_matrix = parse_dir("/home/fra/University/HPPS/Approximate-PR/graph_generator/generated_csc/cur");
    csc_fixed_t fixed_csc = to_fixed_csc(csc_matrix);

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


    // Transform the std::vectors into device vectors
    to_device_csc(d_csc_val, d_csc_non_zero, d_csc_col_idx, fixed_csc);

    std::cout << "Initializing PR, Error, dangling bitmap, update bitmap vecors" << std::endl;

    d_set_value << < MAX_B, MAX_T >> > (d_pr, d_to_fixed(1.0 / DIM), DIM);
    d_set_value << < MAX_B, MAX_T >> > (d_error, d_to_fixed(1.0), DIM);
    d_set_value << < MAX_B, MAX_T >> > (d_dangling_bitmap, true, DIM);
    d_set_value << < MAX_B, MAX_T >> > (d_update_bitmap, true, DIM);

    d_fixed_set_dangling_bitmap << < MAX_B, MAX_T >> > (d_dangling_bitmap, d_csc_col_idx, NON_ZERO);

    // debug_print("d_dangling_bitmap", d_dangling_bitmap, DIM);

    cudaMemcpy(pr, d_pr, DIM * sizeof(num_type), cudaMemcpyDeviceToHost);
    cudaMemcpy(error, d_error, DIM * sizeof(num_type), cudaMemcpyDeviceToHost);

    std::cout << "Beginning pagerank" << std::endl;

    int iterations = 0;
    bool converged = false;

    while (!converged && iterations < MAX_ITER) {

        // SpMV
        d_fixed_spmv << < MAX_B, MAX_T >> > (d_spmv_res, d_pr, d_csc_val, d_csc_non_zero, d_csc_col_idx, DIM);

        // Scale
        d_fixed_scale << < MAX_B, MAX_T >> > (d_spmv_res, d_to_fixed(ALPHA), DIM);

        //TODO: Remember to handle dangling nodes!
        num_type res_v = h_fixed_dot(d_pr, d_dangling_bitmap, DIM);

        // Shift
        d_fixed_shift << < MAX_B, MAX_T >> > (d_spmv_res, d_to_fixed((1.0 - ALPHA) / DIM + ((ALPHA / DIM) * res_v)), DIM);

        // Compute error
        d_fixed_compute_error << < MAX_B, MAX_T >> > (d_error, d_spmv_res, d_pr, DIM);

        cudaDeviceSynchronize();

        cudaMemcpy(error, d_error, DIM * sizeof(num_type), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_pr, d_spmv_res, DIM * sizeof(num_type), cudaMemcpyDeviceToDevice);

        converged = check_error(error, d_to_fixed(TAU), DIM);

        // debug_print("d_pr", d_pr, DIM);
        iterations++;

    }

    cudaMemcpy(pr, d_pr, DIM * sizeof(num_type), cudaMemcpyDeviceToHost);

    std::cout << "Pagerank converged after " << iterations << " iterations" << std::endl;

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
                  if (l.second != r.second)return l.second > r.second;
                  else return l.first > r.first;
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
    int errors = 0;

    while (results >> tmp) {
        if (tmp != sorted_pr_idxs[i]) {
            errors++;
            // std::cout << "ERROR AT INDEX " << i << ": " << tmp << " != " << sorted_pr_idxs[i] << " Value => " << (num_type) pr_map[sorted_pr_idxs[i]] << std::endl;
        }
        i++;
    }

    std::cout << "Percentage of error: " << (((double) errors) / (DIM)) * 100 << "%\n" << std::endl;

    std::cout << "End of computation! Freeing memory..." << std::endl;

    cudaFree(&pr);
    cudaFree(&error);

    cudaFree(&d_pr);
    cudaFree(&d_error);
    cudaFree(&d_spmv_res);
    cudaFree(&d_csc_val);
    cudaFree(&d_csc_non_zero);
    cudaFree(&d_csc_col_idx);

    std::cout << "Done." << std::endl;


    return 0;
}
