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

#define DEBUG true

#define USE_NO_OPTIMIZATION false
#define USE_L2_NORM true
#define USE_L2_NORM_BITMASK false
#define GRAPH_TYPE ((std::string) "smw")

#define PYTHON_PAGERANK_VALUES false 
#define PYTHON_CONVERGENCE_ERROR_OUT false

#define MAX_ITER 200

#define num_type long long unsigned

// 0.000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000
#define SCALE 63

__host__
__device__
__forceinline__
num_type d_to_fixed(double x) {
    return x * ((double) ((num_type) 1 << SCALE));
}

__host__
__device__
__forceinline__
num_type fixed_mult(num_type x, num_type y) {
    return d_to_fixed(((double) ((double) x / (double) (((num_type) 1) << SCALE)) * ((double) y / (double) (((num_type) 1) << SCALE))));
}


csc_fixed_t to_fixed_csc(csc_t m) {

    csc_fixed_t fixed_csc;

    fixed_csc.col_idx = m.col_idx;
    fixed_csc.non_zero = m.non_zero;
    fixed_csc.val = std::vector<num_type>();

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

template<typename T>
__global__
void
d_update_fixed_spmv(T *Y, T *pr, T *csc_val, int *csc_non_zero, int *csc_col_idx, bool *update_bitmap, const int DIMV) {

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const T initial_zero = d_to_fixed(0.0);


    for (int i = init; i < DIMV && update_bitmap[i]; i += stride) {

        int begin = csc_non_zero[i];
        int end = csc_non_zero[i + 1];
        T acc = initial_zero;

        for (int j = begin; j < end; ++j) {
            acc += fixed_mult(csc_val[j], pr[csc_col_idx[j]]);
        }

        Y[i] = acc;

    }
}

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
void d_update_fixed_compute_error(T *error, T *v1, T *v2, bool *update_bitmap, const T max_err, const unsigned DIMV) {

    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = init; i < DIMV && update_bitmap[i]; i += stride) {
        error[i] = d_fixed_abs(v1[i], v2[i]);
        update_bitmap[i] = error[i] >= max_err;
    }

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
        if (e[i] > error) return false;
    }
    return true;
}


template<typename T>
struct d_fixed_add_functor : public thrust::binary_function<T, T, T> {
    __device__
    T operator()(const T &x, const T &y) const {
        return x + y;
    }
};

template<typename T, typename S>
struct d_fixed_mult_functor : public thrust::binary_function<T, S, T> {
    __device__
    T operator()(const T &x, const S &y) const {
        return fixed_mult(x, y);
    }
};

template<typename T1, typename T2>
T2 d_fixed_dot(T1 *x, T2 *y, size_t n) {

    return thrust::inner_product(
            thrust::device,
            thrust::device_pointer_cast(x),
            thrust::device_pointer_cast(x + n),
            thrust::device_pointer_cast(y),
            0,
            d_fixed_add_functor<T2>(),
            d_fixed_mult_functor<T2, T1>()
    );
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
void d_fixed_axpb(T *x, T a, T b, const unsigned DIMV) {
    int init = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = init; i < DIMV; i += stride) {
        x[i] = fixed_mult(x[i], a) + b;
    }

}

struct is_over_error {
    __device__
    bool operator()(num_type &x) {
        return x > d_to_fixed(TAU);
    }
};

struct d_fixed_square_functor {
    __device__
    num_type operator()(num_type &x) {
        return fixed_mult(x, x);
    }
};

template<typename T>
T euclidean_error(T *error, const unsigned DIMV) {
    return thrust::transform_reduce(
            thrust::device,
            error,
            error + DIMV,
            d_fixed_square_functor(),
            0.0,
            d_fixed_add_functor<T>()
    );
}


int main() {

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
    csc_fixed_t fixed_csc = to_fixed_csc(csc_matrix);

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

    // Transform the std::vectors into device vectors
    to_device_csc(d_csc_val, d_csc_non_zero, d_csc_col_idx, fixed_csc);

    if (DEBUG) {
        std::cout << "Initializing PR, Error, dangling bitmap, update bitmap vecors" << std::endl;
    }

    d_set_value << < MAX_B, MAX_T >> > (d_pr, d_to_fixed(1.0 / DIM), DIM);
    d_set_value << < MAX_B, MAX_T >> > (d_error, d_to_fixed(1.0), DIM);
    d_set_value << < MAX_B, MAX_T >> > (d_dangling_bitmap, true, DIM);
    d_set_value << < MAX_B, MAX_T >> > (d_update_bitmap, true, DIM);

    d_fixed_set_dangling_bitmap << < MAX_B, MAX_T >> > (d_dangling_bitmap, d_csc_col_idx, NON_ZERO);

    // debug_print("d_dangling_bitmap", d_dangling_bitmap, DIM);

    cudaMemcpy(pr, d_pr, DIM * sizeof(num_type), cudaMemcpyDeviceToHost);
    cudaMemcpy(error, d_error, DIM * sizeof(num_type), cudaMemcpyDeviceToHost);

    if (DEBUG) {
        std::cout << "Beginning pagerank" << std::endl;
    }

    int iterations = 0;
    bool converged = false;
    const num_type F_ALPHA = d_to_fixed(ALPHA);
    const num_type F_TAU = d_to_fixed(TAU);
    const num_type F_SHIFT = d_to_fixed((1.0 - ALPHA) / DIM);
    const num_type F_DANGLING_SCALE = d_to_fixed(ALPHA / DIM);

    // Start a timer
    auto pr_clock_start = std::chrono::high_resolution_clock::now();

    while (!converged && iterations < MAX_ITER) {

        if(USE_NO_OPTIMIZATION){
            // SpMV
            d_fixed_spmv << < MAX_B, MAX_T >> > (d_spmv_res, d_pr, d_csc_val, d_csc_non_zero, d_csc_col_idx, DIM);
            // Dangling nodes handler
            num_type res_v = d_fixed_dot(d_pr, d_dangling_bitmap, DIM);
            // aX + b
            d_fixed_axpb << < MAX_T, MAX_B >> >(d_spmv_res, F_ALPHA, ((num_type) F_SHIFT + fixed_mult(F_DANGLING_SCALE, res_v)), DIM);
            // Compute error
            d_fixed_compute_error << < MAX_B, MAX_T >> > (d_error, d_spmv_res, d_pr, DIM);

            // Swap back the pagerank values
            cudaMemcpy(d_pr, d_spmv_res, DIM * sizeof(num_type), cudaMemcpyDeviceToDevice);

            // Check for convergence
            converged = thrust::count_if(thrust::device, d_error, d_error + DIM, is_over_error()) == 0;
        }

        if(USE_L2_NORM){
            // SpMV
            d_fixed_spmv << < MAX_B, MAX_T >> > (d_spmv_res, d_pr, d_csc_val, d_csc_non_zero, d_csc_col_idx, DIM);
            // Dangling nodes handler
            num_type res_v = d_fixed_dot(d_pr, d_dangling_bitmap, DIM);
            // aX + b
            d_fixed_axpb << < MAX_T, MAX_B >> >(d_spmv_res, F_ALPHA, ((num_type) F_SHIFT + fixed_mult(F_DANGLING_SCALE, res_v)), DIM);
            // Compute error
            d_fixed_compute_error << < MAX_B, MAX_T >> > (d_error, d_spmv_res, d_pr, DIM);

            // Compute the l2 norm
            num_type error_euc = euclidean_error(d_error, DIM);
            //convergence_error_vector[iterations] = error_euc;

            // Swap back the pagerank values
            cudaMemcpy(d_pr, d_spmv_res, DIM * sizeof(num_type), cudaMemcpyDeviceToDevice);

            // Check for convergence
            converged = error_euc <= F_TAU;
        }

        if(USE_L2_NORM_BITMASK){
            // SpMV
            d_update_fixed_spmv<< <MAX_B, MAX_T>> > (d_spmv_res, d_pr, d_csc_val, d_csc_non_zero, d_csc_col_idx, d_update_bitmap, DIM);
            // Dangling nodes handler
            num_type res_v = d_fixed_dot(d_pr, d_dangling_bitmap, DIM);
            // aX + b
            d_fixed_axpb << < MAX_T, MAX_B >> >(d_spmv_res, F_ALPHA, ((num_type) F_SHIFT + fixed_mult(F_DANGLING_SCALE, res_v)), DIM);
            // Compute error and bitmask
            d_update_fixed_compute_error << <MAX_B, MAX_T>> > (d_error, d_spmv_res, d_pr, d_update_bitmap, F_TAU, DIM);

            // Compute the l2 norm
            num_type error_euc = euclidean_error(d_error, DIM);
            // convergence_error_vector[iterations] = error_euc;

            // Swap back the pagerank values
            cudaMemcpy(d_pr, d_spmv_res, DIM * sizeof(num_type), cudaMemcpyDeviceToDevice);

            // Check for convergence
            converged = error_euc <= F_TAU;
        }

/*
        // SpMV
        d_fixed_spmv << < MAX_B, MAX_T >> > (d_spmv_res, d_pr, d_csc_val, d_csc_non_zero, d_csc_col_idx, DIM);
        //d_update_fixed_spmv<< <MAX_B, MAX_T>> > (d_spmv_res, d_pr, d_csc_val, d_csc_non_zero, d_csc_col_idx, d_update_bitmap, DIM);

        // Dangling nodes handler
        num_type res_v = d_fixed_dot(d_pr, d_dangling_bitmap, DIM);
        //num_type res_v = h_fixed_dot(DIM, d_dangling_bitmap, d_pr);
        //std::cout << "Thrust: " << res_v << " <-> Host: " << res_v_h << " -> diff: " << h_s_abs(res_v_h, res_v) << std::endl;

        // aX + b
        d_fixed_axpb << < MAX_T, MAX_B >> >(d_spmv_res, F_ALPHA, ((num_type) F_SHIFT + fixed_mult(F_DANGLING_SCALE, res_v)), DIM);

        // Compute error
        d_fixed_compute_error << < MAX_B, MAX_T >> > (d_error, d_spmv_res, d_pr, DIM);
        //d_update_fixed_compute_error << <MAX_B, MAX_T>> > (d_error, d_spmv_res, d_pr, d_update_bitmap, F_TAU, DIM);
        num_type error_euc = euclidean_error(d_error, DIM);
        convergence_error_vector[iterations] = error_euc;

        cudaMemcpy(d_pr, d_spmv_res, DIM * sizeof(num_type), cudaMemcpyDeviceToDevice);

        //converged = thrust::count_if(thrust::device, d_error, d_error + DIM, is_over_error()) == 0;
        converged = error_euc <= F_TAU;*/
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
        std::cout << "Pagerank converged after " << iterations << " iterations" << std::endl;
    }

    std::map<int, num_type> pr_map;
    std::vector<std::pair<int, num_type>> sorted_pr;
    std::vector<int> sorted_pr_idxs;

    for (int i = 0; i < DIM; ++i) {
        sorted_pr.push_back({i, pr[i]});
        pr_map[i] = pr[i];
    }


    std::sort(sorted_pr.begin(), sorted_pr.end(),
              [](const std::pair<int, num_type> &l, const std::pair<int, num_type> &r) {
                  if (l.second != r.second)return l.second > r.second;
                  else return l.first > r.first;
              });

    for (auto const &pair: sorted_pr) {
        sorted_pr_idxs.push_back(pair.first);
        //std::cout << pair.first << "," << pair.second << std::endl;
    }
    if (DEBUG) {
        std::cout << "Checking results..." << std::endl;

        std::ifstream results;
        results.open("/home/fra/University/HPPS/Approximate-PR/new_ds/" + GRAPH_TYPE + "/results.txt");

        int i = 0;
        int tmp = 0;
        int errors = 0;

        int prev_left_idx = 0;
        int prev_right_idx = 0;

        while (results >> tmp) {
            if (tmp != sorted_pr_idxs[i]) {
                if (prev_left_idx != sorted_pr_idxs[i] || prev_right_idx != tmp) {
                    errors++;
                    if (errors <= 10) {
                        // Print only the top 10 errors
                        std::cout << "ERROR AT INDEX " << i << ": " << tmp << " != " << sorted_pr_idxs[i]
                                  << " Value => " << (num_type) pr_map[sorted_pr_idxs[i]] << std::endl;
                    }
                }

                prev_left_idx = tmp;
                prev_right_idx = sorted_pr_idxs[i];

            }
            i++;
        }

        std::cout << "Percentage of error: " << (((double) errors) / (DIM)) * 100 << "%\n" << std::endl;

        std::cout << "End of computation! Freeing memory..." << std::endl;
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
