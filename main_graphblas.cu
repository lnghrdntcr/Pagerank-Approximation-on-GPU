//
// Created by Francesco Sgherzi on 25/05/19.
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



#define GRB_USE_CUDA
#include "graphblas/graphblas.hpp"

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

int main() {

    return 0;
}

