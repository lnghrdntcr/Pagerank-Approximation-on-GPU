// Created by Francesco Sgherzi on 15/04/19.
//

#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>

#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

#include "Parse/Parse.h"
#include "Utils/Utils.h"

#define TAU 0.0
#define ALPHA 0.85

#define MAX_B 1024
#define MAX_T 1024

#define MAX_ITER 200

#define num_type double

#define DEBUG true

#define USE_NO_OPTIMIZATION false
#define USE_L2_NORM true
#define USE_L2_NORM_BITMASK false
#define GRAPH_TYPE ((std::string) "gnp")

#define PYTHON_PAGERANK_VALUES false
#define PYTHON_CONVERGENCE_ERROR_OUT false

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

    for (int i = init; i < DIMV && update_bitmap[i]; i += stride) {

        int begin = csc_non_zero[i];
        int end = csc_non_zero[i + 1];
        T acc = 0.0;

        for (int j = begin; j < end; j++) {
            acc += csc_val[j] * pr[csc_col_idx[j]];
        }

        Y[i] = acc;

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

/**
 * Performs an axpb operation on the x vector inplace
 * @tparam T Numeric type
 * @param x The vector to scale and shift
 * @param a scaling factor
 * @param b shifting factor
 * @return
 */
template<typename T>
__global__
void axpb(T *x, T a, T b, const unsigned DIMV) {
    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = init; i < DIMV; i += stride) {
        x[i] = x[i] * a + b;
    }

}

template<typename T>
struct euclidean_functor : public thrust::binary_function<T, T, T> {
    __device__
    T operator()(const T &x, const T &y) const {
        return (x - y) * (x - y);
    }
};

// Compute Euclidean norm of the difference of 2 vectors;
template<typename T>
T euclidean_dist(size_t n, T *x, T *y) {
    return std::sqrt(thrust::inner_product(
            thrust::device, thrust::device_pointer_cast(x),
            thrust::device_pointer_cast(x + n), thrust::device_pointer_cast(y),
            0.0f, thrust::plus<T>(), euclidean_functor<T>()));
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

    for (int i = init; i < DIMV && update_bitmap[i]; i += stride) {
        error[i] = abs(next[i] - prev[i]);
        update_bitmap[i] = error[i] >= TAU;
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

    return thrust::inner_product(thrust::device_pointer_cast(x),
                                 thrust::device_pointer_cast(x + n),
                                 thrust::device_pointer_cast(y), 0.0);

}

struct is_over_error {
    __device__
    bool operator()(num_type &x) {
        return x > TAU;
    }
};

struct d_square : public thrust::unary_function<num_type, num_type> {
    __device__
    num_type operator()(num_type &x) {
        return x * x;
    }
};

template<typename T>
T euclidean_error(T *error, const unsigned DIMV) {
    thrust::plus <T> add;
    return thrust::transform_reduce(
            thrust::device,
            error,
            error + DIMV,
            d_square(),
            0.0,
            add
    );
}

template<typename T1, typename T2>
T2 h_dot(size_t n, T1 *x, T2 *y) {
    T1 *tempx;
    T2 *tempy;

    cudaMallocHost(&tempx, sizeof(T1) * n);
    cudaMallocHost(&tempy, sizeof(T2) * n);

    cudaMemcpy(tempx, x, n * sizeof(T1), cudaMemcpyDeviceToHost);
    cudaMemcpy(tempy, y, n * sizeof(T1), cudaMemcpyDeviceToHost);

    T2 acc = 0.0;

    for (int i = 0; i < n; ++i) {
        acc += tempx[i] * tempy[i];
    }

    return acc;
}

int main() {
    cudaDeviceReset();

    /**
     * HOST
     */
    num_type *pr;
    num_type *error;
    num_type *convergence_error_vector;

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

    csc_t csc_matrix = parse_dir("/home/fra/University/HPPS/Approximate-PR/new_ds/" + GRAPH_TYPE, DEBUG);

    const unsigned NON_ZERO = csc_matrix.val.size();
    const unsigned DIM = csc_matrix.non_zero.size() - 1;

    if (DEBUG) {
        std::cout << "\nFEATURES: " << std::endl;
        std::cout << "\tNumber of non zero elements: " << NON_ZERO << std::endl;
        std::cout << "\tNumber of nodes: " << DIM << std::endl;
        std::cout << "\tSparseness: " << (1 - (((double) NON_ZERO) / (DIM * DIM))) * 100 << "%\n" << std::endl;
    }

    cudaMallocHost(&pr, sizeof(num_type) * DIM);
    cudaMallocHost(&error, sizeof(num_type) * DIM);

    if (DEBUG) {
        std::cout << "Initializing device memory" << std::endl;
    }

    // Create device memory
    cudaMalloc(&d_csc_val, sizeof(num_type) * NON_ZERO);
    cudaMalloc(&d_csc_non_zero, sizeof(int) * (DIM + 1));
    cudaMalloc(&d_csc_col_idx, sizeof(num_type) * NON_ZERO);
    cudaMalloc(&d_pr, sizeof(num_type) * DIM);
    cudaMalloc(&d_error, sizeof(num_type) * DIM);
    cudaMalloc(&d_spmv_res, sizeof(num_type) * DIM);
    cudaMalloc(&d_dangling_bitmap, DIM * sizeof(bool));
    cudaMalloc(&d_update_bitmap, DIM * sizeof(bool));

    convergence_error_vector = (num_type *) calloc(MAX_ITER, sizeof(num_type));

    if (DEBUG) {
        std::cout << "Parsing csc files" << std::endl;
    }

    to_device_csc(d_csc_val, d_csc_non_zero, d_csc_col_idx, csc_matrix);

    if (DEBUG) {
        std::cout << "Initializing pr, error, dangling bitmap vectors" << std::endl;
    }

    // Initialize error and pr vector
    cudaMemset(d_pr, (num_type) 1.0 / DIM, DIM);
    cudaMemset(d_error, (num_type) 1.0, DIM);
    cudaMemset(d_dangling_bitmap, true, DIM);
    cudaMemset(d_update_bitmap, true, DIM);

    d_set_dangling_bitmap << < MAX_B, MAX_T >> > (d_dangling_bitmap, d_csc_col_idx, NON_ZERO);

    // Copy them back to their host vectors
    cudaMemcpy(pr, d_pr, DIM * sizeof(num_type), cudaMemcpyDeviceToHost);
    cudaMemcpy(error, d_error, DIM * sizeof(num_type), cudaMemcpyDeviceToHost);

    if (DEBUG) {
        std::cout << "Beginning pagerank..." << std::endl;
    }

    int iterations = 0;
    bool converged = false;

    auto pr_clock_start = std::chrono::high_resolution_clock::now();

    while (!converged && iterations < MAX_ITER) {

        if(USE_NO_OPTIMIZATION){
            // SpMV
            spmv << < MAX_B, MAX_T >> > (d_spmv_res, d_pr, d_csc_val, d_csc_non_zero, d_csc_col_idx, DIM);
            // Dangling nodes handler
            num_type res_v = dot(DIM, d_pr, d_dangling_bitmap);
            // aX + b

            axpb <<< MAX_B, MAX_T >>> (
                    d_spmv_res,
                            (num_type) ALPHA,
                            static_cast<num_type>((1.0 - ALPHA) / DIM + (ALPHA / DIM) * res_v),
                            DIM
            );

            // Compute error
            compute_error << < MAX_B, MAX_T >> > (d_error, d_spmv_res, d_pr, DIM);

            // Swap back the pagerank values
            cudaMemcpy(d_pr, d_spmv_res, DIM * sizeof(num_type), cudaMemcpyDeviceToDevice);

            // Check for convergence
            converged = thrust::count_if(thrust::device, d_error, d_error + DIM, is_over_error()) == 0;
        }

        if(USE_L2_NORM){
            // SpMV
            spmv << < MAX_B, MAX_T >> > (d_spmv_res, d_pr, d_csc_val, d_csc_non_zero, d_csc_col_idx, DIM);
            // Dangling nodes handler
            num_type res_v = dot(DIM, d_pr, d_dangling_bitmap);
            // aX + b

            axpb <<< MAX_B, MAX_T >>> (
                    d_spmv_res,
                            (num_type) ALPHA,
                            static_cast<num_type>((1.0 - ALPHA) / DIM + (ALPHA / DIM) * res_v),
                            DIM
            );

            // Compute error
            compute_error << < MAX_B, MAX_T >> > (d_error, d_spmv_res, d_pr, DIM);

            // Compute the l2 norm
            num_type error_euc = euclidean_error(d_error, DIM);
            convergence_error_vector[iterations] = error_euc;

            // Swap back the pagerank values
            cudaMemcpy(d_pr, d_spmv_res, DIM * sizeof(num_type), cudaMemcpyDeviceToDevice);

            // Check for convergence
            converged = error_euc <= TAU;
        }

        if(USE_L2_NORM_BITMASK){
            // SpMV
            part_spmv<< <MAX_B, MAX_T>> > (d_spmv_res, d_pr, d_csc_val, d_csc_non_zero, d_csc_col_idx, d_update_bitmap, DIM);
            // Dangling nodes handler
            num_type res_v = dot(DIM, d_pr, d_dangling_bitmap);
            // aX + b

            axpb <<< MAX_B, MAX_T >>> (
                    d_spmv_res,
                            (num_type) ALPHA,
                            static_cast<num_type>((1.0 - ALPHA) / DIM + (ALPHA / DIM) * res_v),
                            DIM
            );

            // Compute error and bitmask
            part_compute_error << < MAX_B, MAX_T >> > (d_error, d_spmv_res, d_pr, d_update_bitmap, DIM);

            // Compute the l2 norm
            num_type error_euc = euclidean_error(d_error, DIM);
            // convergence_error_vector[iterations] = error_euc;

            // Swap back the pagerank values
            cudaMemcpy(d_pr, d_spmv_res, DIM * sizeof(num_type), cudaMemcpyDeviceToDevice);

            // Check for convergence
            converged = error_euc <= TAU;
        }
        /*
        spmv << < MAX_B, MAX_T >> > (d_spmv_res, d_pr, d_csc_val, d_csc_non_zero, d_csc_col_idx, DIM);
        //part_spmv << < MAX_B, MAX_T >> > (d_spmv_res, d_pr, d_csc_val, d_csc_non_zero, d_csc_col_idx, d_update_bitmap, DIM);

        num_type res_v = dot(DIM, d_dangling_bitmap, d_pr);

        axpb <<< MAX_B, MAX_T >>> (
                d_spmv_res,
                (num_type) ALPHA,
                static_cast<num_type>((1.0 - ALPHA) / DIM + (ALPHA / DIM) * res_v),
                DIM
        );

        //num_type euclidean_error = euclidean_dist(DIM, d_error, d_pr);
        compute_error << < MAX_B, MAX_T >> > (d_error, d_spmv_res, d_pr, DIM);
        //part_compute_error << < MAX_B, MAX_T >> > (d_error, d_spmv_res, d_pr, d_update_bitmap, DIM);

        num_type error_euc = euclidean_error(d_error, DIM);
        convergence_error_vector[iterations] = error_euc;
        //std::cout << "Convergence error[" << iterations << "]: " << error_euc << std::endl;

        cudaMemcpy(d_pr, d_spmv_res, DIM * sizeof(num_type), cudaMemcpyDeviceToDevice);

        //converged = thrust::count_if(thrust::device, d_error, d_error + DIM, is_over_error()) == 0;
        converged = error_euc <= TAU;
        */
        iterations++;
    }

    // Stop the timer
    auto pr_clock_end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(pr_clock_end - pr_clock_start).count();

    if (DEBUG) {
        std::cout << "Pagerank converged after " << duration << " ms" << std::endl;
    }

    cudaMemcpy(pr, d_pr, DIM * sizeof(num_type), cudaMemcpyDeviceToHost);


    if (DEBUG) {
        std::cout << "converged after n_iter: " << iterations << std::endl;
    }

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

    if (DEBUG) {
        std::cout << "Checking results..." << std::endl;

        std::ifstream results;
        results.open("/home/fra/University/HPPS/Approximate-PR/new_ds/" + GRAPH_TYPE + "/results.txt");

        int i = 0;
        int tmp = 0;
        int errors = 0;
        int errors_real = 0;

        int prev_left_idx = 0;
        int prev_right_idx = 0;

        while (results >> tmp) {
            // std::cout << "reading " << tmp << std::endl;
            if (tmp != sorted_pr_idxs[i]) {
                errors_real++;
                if (prev_left_idx != sorted_pr_idxs[i] || prev_right_idx != tmp) {
                    errors++;

                    if (errors <= 10) {
                        // Print only the top 10 errors
                        std::cout << "ERROR AT INDEX " << i << ": " << tmp << " != " << sorted_pr_idxs[i]
                                  << " Value => "
                                  << (num_type) pr_map[sorted_pr_idxs[i]] << std::endl;
                    }

                }

                prev_left_idx = tmp;
                prev_right_idx = sorted_pr_idxs[i];


            }
            i++;
        }

        std::cout << "Percentage of error: " << (((double) errors_real) / (DIM)) * 100 << "%\n" << std::endl;


    }

    if (PYTHON_CONVERGENCE_ERROR_OUT) {
        for (int i = 0; i < iterations; ++i) {
            std::cout << "(" << i << "," << convergence_error_vector[i] << ")" << std::endl;
        }
    }

    if (PYTHON_PAGERANK_VALUES) {
        for (auto const &pair: sorted_pr) {
            std::cout << pair.first << "," << pair.second << std::endl;
        }
    }


    cudaFree(&pr);
    cudaFree(&error);

    cudaFree(&d_pr);
    cudaFree(&d_error);
    cudaFree(&d_spmv_res);
    cudaFree(&d_csc_val);
    cudaFree(&d_csc_non_zero);
    cudaFree(&d_csc_col_idx);

    cudaDeviceReset();

    return 0;
}
