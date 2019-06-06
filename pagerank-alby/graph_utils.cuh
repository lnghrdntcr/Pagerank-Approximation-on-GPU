#pragma once

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

////////////////////////////////
////////////////////////////////

#define DEBUG 1

// Error check;
#undef cudaCheckError
#ifdef DEBUG
#define WHERE " at: " << __FILE__ << ':' << __LINE__
#define cudaCheckError()                                                    \
    {                                                                       \
        cudaError_t e = cudaGetLastError();                                 \
        if (e != cudaSuccess) {                                             \
            std::cerr << "Cuda failure: " << cudaGetErrorString(e) << WHERE \
                      << std::endl;                                         \
        }                                                                   \
    }
#else
#define cudaCheckError()
#define WHERE ""
#endif

////////////////////////////////
////////////////////////////////

// Store "value" on n values of an input vector x;
template <typename T>
void fill(size_t n, T *x, T value, cudaStream_t const &stream = 0) {
    if (stream) {
        thrust::fill(thrust::cuda::par.on(stream),
                     thrust::device_pointer_cast(x),
                     thrust::device_pointer_cast(x + n), value);
    } else {
        thrust::fill(thrust::device_pointer_cast(x),
                     thrust::device_pointer_cast(x + n), value);
    }

    cudaCheckError();
}

// Dot product of 2 vectors;
template <typename T1, typename T2>
T2 dot(size_t n, T1 *x, T2 *y, cudaStream_t const &stream = 0) {
    T2 result;
    if (stream) {
        result = thrust::inner_product(thrust::cuda::par.on(stream),
                                       thrust::device_pointer_cast(x),
                                       thrust::device_pointer_cast(x + n),
                                       thrust::device_pointer_cast(y), 0.0f);
    } else {
        result = thrust::inner_product(thrust::device_pointer_cast(x),
                                       thrust::device_pointer_cast(x + n),
                                       thrust::device_pointer_cast(y), 0.0f);
    }
    cudaCheckError();
    return result;
}

// Dot product of 2 vectors, with one being a boolean vector
template <typename T>
T filter_sum(size_t n, bool *x, T *y) {
    T result = thrust::inner_product(
        thrust::device_pointer_cast(x), thrust::device_pointer_cast(x + n),
        thrust::device_pointer_cast(y), 0.0f, thrust::plus<T>(),
        thrust::logical_and<T>());
    cudaCheckError();
    return result;
}

template <typename T>
struct axpy_c_functor : public thrust::unary_function<T, T> {
    const T a;
    const T y;
    axpy_c_functor(T _a, T _y) : a(_a), y(_y) {}

    __host__ __device__ T operator()(const T &x) const { return a * x + y; }
};

// x = a * x + y, with y being a constant value;
template <typename T>
void axpy_c(size_t n, T a, T *x, T y) {
    thrust::transform(thrust::device_pointer_cast(x),
                      thrust::device_pointer_cast(x + n),
                      thrust::device_pointer_cast(x), axpy_c_functor<T>(a, y));
    cudaCheckError();
}

template <typename T>
struct euclidean_functor : public thrust::binary_function<T, T, T> {
    __host__ __device__ T operator()(const T &x, const T &y) const {
        return (x - y) * (x - y);
    }
};

// Compute Euclidean norm of the difference of 2 vectors;
template <typename T>
T euclidean_dist(size_t n, T *x, T *y, cudaStream_t const &stream = 0) {
    T result;
    if (stream) {
        result = std::sqrt(thrust::inner_product(
            thrust::cuda::par.on(stream), thrust::device_pointer_cast(x),
            thrust::device_pointer_cast(x + n), thrust::device_pointer_cast(y),
            0.0f, thrust::plus<T>(), euclidean_functor<T>()));
    } else {
        result = std::sqrt(thrust::inner_product(
            thrust::cuda::par.on(stream), thrust::device_pointer_cast(x),
            thrust::device_pointer_cast(x + n), thrust::device_pointer_cast(y),
            0.0f, thrust::plus<T>(), euclidean_functor<T>()));
    }
    cudaCheckError();
    return result;
}

// Compute Euclidean norm of the difference of 2 vectors, on CPU;
template <typename T>
T euclidean_dist_cpu(size_t n, T *v1, T *v2) {
    T norm = 0;
#pragma omp parallel for reduction(+ \
                                   : norm)
    for (int i = 0; i < n; i++) {
        T diff = v1[i] - v2[i];
        norm += +diff * diff;
    }
    return sqrt(norm);
}

////////////////////////////////
////////////////////////////////
