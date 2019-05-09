// Created by Francesco Sgherzi on 15/04/19.
//

// TODO: Check graphblast
// TODO: Implement spmv with bitmask graphblast
// TODO: Different data types.
// TODO: multiply pr value by int max to use fixed point (?)

#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef __clang__
#include <cuda_builtin_vars.h>
#endif

#include <thrust/inner_product.h>

#include "Parse/Parse.h"
#include "Utils/Utils.h"

#define TAU 0.0
#define ALPHA 0.85

#define MAX_B 1024
#define MAX_T 1024

#define MAX_ITER 200

#define num_type unsigned

#define SCALE 30
// 00.00 0000 0000 0000 0000 0000 0000 0000

#define d_to_fixed(x) ((x) * ( (double) (1 << SCALE)))
#define fixed_mult(x, y) ((((x) >> SCALE / 2) * ((y) >> SCALE / 2)) >> 0)
#define fixed_to_d(x) ((double)(x) / (double)(1 << SCALE))

// TODO: Figure out a way to make fixed point division work
// #define fixed_div(x, y) ((x << SCALE / 2) / (y << SCALE / 2))

csc_fixed_t to_fixed_csc(csc_t m){

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
    int      *d_csc_non_zero;
    int      *d_csc_col_idx;
    bool     *d_dangling_bitmap;
    bool     *d_update_bitmap;

    csc_t csc_matrix = parse_dir("/home/fra/University/HPPS/Approximate-PR/graph_generator/generated_csc/test");
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


    to_device_csc(d_csc_val, d_csc_non_zero, d_csc_col_idx, fixed_csc);




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
