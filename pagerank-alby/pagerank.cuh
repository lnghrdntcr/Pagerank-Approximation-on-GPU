#pragma once

#include <cuda_runtime.h>

#define NUM_STREAMS 3

template <typename IndexType, typename ValueType>
int pagerank(
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
    ValueType *final_error,
    bool human_readable = false,
    bool error_on_cpu = false);

template <typename IndexType, typename ValueType>
bool pagerank_iteration(
    IndexType n,           // |V|
    IndexType e,           // |E|
    IndexType *csc_ptr,    // Cumulative number (moving across each column) of non-zero entry of the adjacency matrix, stored as CSC for fast in-neighbourhood access;
    IndexType *csc_ind,    // For each non-zero entry, row number where it appears;
    ValueType *csc_val,    // Non-zero values of the adjacency matrix;
    ValueType alpha,       // Damping factor;
    bool *dangling_bitmap, // vector of size |V|, for each vertex value=1 if dangling, 0 if not dangling;
    float tolerance,
    int iter,
    int max_iter,
    ValueType *&tmp,          // Vector which contains temporary PR values, which is modified during each iteration;
    void *cub_d_temp_storage, // Temporary device storage for PR values;
    size_t cub_temp_storage_bytes,
    ValueType *&pr, // Vector where PR values are stored;
    ValueType *final_error,
    cudaStream_t *streams, // Vector of streams
    bool human_readable = false,
    bool error_on_cpu = false);
