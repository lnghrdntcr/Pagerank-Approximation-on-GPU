#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>

template <typename ValueType>
void cpu_controller(ValueType *&tmp, ValueType *&pr, float tolerance, float *final_error, int n, int max_it, bool human_readable = false);

template <typename IndexType, typename ValueType>
void gpu_controller(
    IndexType n,           // |V|
    IndexType e,           // |E|
    IndexType *csc_ptr,    // Cumulative number (moving across each column) of non-zero entry of the adjacency matrix, stored as CSC for fast in-neighbourhood access;
    IndexType *csc_ind,    // For each non-zero entry, row number where it appears;
    ValueType *csc_val,    // Non-zero values of the adjacency matrix;
    ValueType alpha,       // Damping factor;
    bool *dangling_bitmap, // vector of size |V|, for each vertex value=1 if dangling, 0 if not dangling;
    int max_it,
    ValueType *&tmp,          // Vector which contains temporary PR values, which is modified during each iteration;
    void *cub_d_temp_storage, // Temporary device storage for PR values;
    size_t cub_temp_storage_bytes,
    ValueType *&pr, // Vector where PR values are stored;
    int *final_iteration,
    bool human_readable = false);

template <typename IndexType, typename ValueType>
int pagerank_pipeline(
    IndexType n,        // |V|
    IndexType e,        // |E|
    IndexType *csc_ptr, // Cumulative number (moving across each column) of non-zero entry of the adjacency matrix, stored as CSC for fast in-neighbourhood access;
    IndexType *csc_ind, // For each non-zero entry, row number where it appears;
    ValueType *csc_val, // Non-zero values of the adjacency matrix;
    ValueType alpha,    // Damping factor;
    float tolerance,
    int max_iter,
    ValueType *pagerank_vector, // Initial values of PR. Pass a reference to a pointer, because we swap "pr" with "tmp" after each iteration. If we passed *pr, we could modify its content, but not where the pointer "points" (changes would be local, the address is passed by copy);
    int *final_iteration,
    float *final_error,
    bool human_readable = false);

template <typename IndexType, typename ValueType>
void pagerank_iteration(
    IndexType n,              // |V|
    IndexType e,              // |E|
    IndexType *csc_ptr,       // Cumulative number (moving across each column) of non-zero entry of the adjacency matrix, stored as CSC for fast in-neighbourhood access;
    IndexType *csc_ind,       // For each non-zero entry, row number where it appears;
    ValueType *csc_val,       // Non-zero values of the adjacency matrix;
    ValueType alpha,          // Damping factor;
    bool *dangling_bitmap,    // vector of size |V|, for each vertex value=1 if dangling, 0 if not dangling;
    ValueType *&tmp,          // Vector which contains temporary PR values, which is modified during each iteration;
    void *cub_d_temp_storage, // Temporary device storage for PR values;
    size_t cub_temp_storage_bytes,
    ValueType *&pr // Vector where PR values are stored;
);

template <typename ValueType>
bool check_convergence(ValueType *&tmp, ValueType *&pr, float tolerance, float *final_error, int n);
