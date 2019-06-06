#include <chrono>
#include <cub/cub.cuh>
#include <stdio.h>
#include <stdlib.h>
#include <thread>

#include "graph_utils.cuh"
#include "pagerank.cuh"
#include "utils.hpp"

////////////////////////////////
////////////////////////////////

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

template <typename IndexType, typename ValueType>
int pagerank(
    IndexType n,
    IndexType e,
    IndexType *csc_ptr,
    IndexType *csc_ind,
    ValueType *csc_val,
    ValueType alpha,
    float tolerance,
    int max_iter,
    ValueType *pagerank_vector,
    int *final_iteration,
    ValueType *final_error,
    bool human_readable,
    bool error_on_cpu) {

    auto start_pr = clock_type::now();

    int max_it = max_iter > 0 ? max_iter : 100;                                // Maximum number of iterations;
    int i = 0;                                                                 // Current iteration;
    float tol = (tolerance < 1.0f && tolerance >= 0.0f) ? tolerance : 1.0E-6f; // Tolerance;
    bool converged = false;                                                    // True if PR has converged;
    ValueType random_probability = static_cast<ValueType>(1.0 / n);            // Initial values of PR;

    if (alpha <= 0.0f || alpha >= 1.0f) {
        return -1;
    }

    bool *dangling_bitmap = 0; // Bitmap with values = 1 for dangling vertices;
    ValueType *tmp = 0;        // Temporary PageRank values;
    ValueType *pr_mng = 0;

    // Temporary storage allocated on GPU;
    void *cub_d_temp_storage = NULL;
    size_t cub_temp_storage_bytes = 0;

    // Create CUDA streams;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++)
        cudaStreamCreate(&streams[i]);
    cudaCheckError();

    // Allocates unified host-device memory;
    cudaMallocManaged(&dangling_bitmap, sizeof(bool) * n);
    cudaMallocManaged(&tmp, sizeof(ValueType) * n);
    cudaMallocManaged(&pr_mng, sizeof(ValueType) * n);

    // Turn the input CSC into managed memory;
    IndexType *csc_ptr_mng = 0;
    IndexType *csc_ind_mng = 0;
    ValueType *csc_val_mng = 0;
    cudaMallocManaged(&csc_ptr_mng, sizeof(IndexType) * (n + 1));
    cudaMallocManaged(&csc_ind_mng, sizeof(IndexType) * e);
    cudaMallocManaged(&csc_val_mng, sizeof(ValueType) * e);
    std::thread t_a(memcpy, csc_ptr_mng, csc_ptr, sizeof(IndexType) * (n + 1));
    std::thread t_b(memcpy, csc_ind_mng, csc_ind, sizeof(IndexType) * e);
    std::thread t_c(memcpy, csc_val_mng, csc_val, sizeof(ValueType) * e);
    t_a.join();
    t_b.join();
    t_c.join();
    cudaCheckError();

    auto start_pr_preprocessing = clock_type::now();

    // Initialize PageRank values with 1/N;
    fill(n, pr_mng, random_probability, streams[0]);
    fill(n, tmp, random_probability, streams[1]);

    // "Fake" matrix multiplication, used to compute the size of cub_d_temp_storage and allocate it.
    // If cub_d_temp_storage is NULL no matrix multiplication is done, but it is used only for initialization.
    // Size of temporary storage is saved in cub_temp_storage_bytes;
    cub::DeviceSpmv::CsrMV(cub_d_temp_storage, cub_temp_storage_bytes, csc_val_mng,
                           csc_ptr_mng, csc_ind_mng, tmp, pr_mng, n, n, e, streams[2]);
    // Allocate temporary storage on GPU;
    cudaMallocManaged(&cub_d_temp_storage, cub_temp_storage_bytes);
    cudaCheckError();

    // Set the bitmap to 1 (assume all vertices to be dangling);
    memset(dangling_bitmap, 1, sizeof(bool) * n);

    // Divide each edge weight (equal to 1) by the out-degree of its source;
    int *outdegree = 0;
    outdegree = (int *)calloc(n, sizeof(int));
    for (int i = 0; i < e; i++) {
        outdegree[csc_ind[i]]++;
    }

#pragma omp parallel for
    for (int i = 0; i < e; i++) {
        csc_val_mng[i] /= outdegree[csc_ind[i]]; // Divide each edge weight by the outdegree of its source vertex;
        dangling_bitmap[csc_ind[i]] = 0;         // Vertices with at least outdegree = 1 are not dangling;
    }

    auto end_pr_preprocessing = clock_type::now();
    auto start_pr_main = clock_type::now();

    // Main PR iterations;
    if (human_readable) {
        std::cout << "- Starting PR Computation ----" << std::endl;
    }
    while (!converged && i < max_it) {
        i++;
        *final_iteration = i;
        converged = pagerank_iteration(
            n,
            e,
            csc_ptr_mng,
            csc_ind_mng,
            csc_val_mng,
            alpha,
            dangling_bitmap,
            tol,
            i,
            max_it,
            tmp,
            cub_d_temp_storage,
            cub_temp_storage_bytes,
            pr_mng,
            final_error,
            streams,
            human_readable,
            error_on_cpu);
    }
    cudaDeviceSynchronize();

    auto end_pr_main = clock_type::now();

    // Copy PR results to output;
    memcpy(pagerank_vector, pr_mng, sizeof(ValueType) * n);
    // std::cout << "------------------------------" << std::endl;
    // std::cout << "- PageRank Values:" << std::endl;
    // print_array_indexed(pagerank_vector, n);
    // std::cout << "------------------------------" << std::endl;

    // Release memory;
    cudaFree(csc_ptr_mng);
    cudaFree(csc_ind_mng);
    cudaFree(csc_val_mng);
    cudaFree(tmp);
    cudaFree(cub_d_temp_storage);
    cudaFree(dangling_bitmap);
    cudaFree(pr_mng);

    if (human_readable) {
        std::cout << "------------------------------" << std::endl;
        std::cout << "- Number of Iterations: " << *final_iteration << std::endl;
        std::cout << "- Convergence Error: " << *final_error << std::endl;
        std::cout << "------------------------------" << std::endl;
    }

    auto end_pr = clock_type::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_pr - start_pr).count();
    auto duration_preproc = chrono::duration_cast<chrono::milliseconds>(end_pr_preprocessing - start_pr_preprocessing).count();
    auto duration_main = chrono::duration_cast<chrono::milliseconds>(end_pr_main - start_pr_main).count();
    if (human_readable) {
        std::cout << "- PR Exec. Time: " << duration << " ms;" << std::endl;
        std::cout << "- PR Preprocessing: " << duration_preproc << " ms;" << std::endl;
        std::cout << "- PR Main Computation: " << duration_main << " ms;" << std::endl;
        std::cout << "------------------------------" << std::endl;
    } else {
        std::string pr_type = error_on_cpu ? "cpu+gpu" : "gpu";
        std::cout << "Pagerank converged after " << duration  << " ms and " << i << " iterations" << std::endl;
    }

    return converged ? 0 : 1;
}

////////////////////////////////
////////////////////////////////

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
    cudaStream_t *streams,
    bool human_readable,
    bool error_on_cpu // If true, compute the error term on CPU
) {
    // Dangling factor, multiply current PR by dangling bitmap;
    ValueType dot_res = dot(n, dangling_bitmap, pr, streams[0]);
    // Compute the "extra" factor given to all PR values, i.e. dangling and teleport factors;
    ValueType extra = (alpha / n) * dot_res + (1 - alpha) / n;
    // Compute a PR iteration as a y = A*x sparse matrix multiplication;
    // Note that cub_d_temp_storage and tmp are allocated with unified memory (the other vectors are inputs to the function, but they are probably also allocated in unified memory)
    // Important remark: we do a CSR multiplication using a CSC, as reading a CSC as CSR is equivalent to transposing the CSC;
    cub::DeviceSpmv::CsrMV(
        cub_d_temp_storage,     // Temporary storage used during the multiplication;
        cub_temp_storage_bytes, // Size of temporary storage;
        csc_val,                // Row-size normalized adjacency matrix A, as CSC;
        csc_ptr,
        csc_ind,
        pr,  // x vector, it contains the current PR values;
        tmp, // y vector, where the new PR values are stored;
        n,   // Number of rows of A;
        n,   // Number of columns of A;
        e,   // Non-zero elements in A, i.e. |E|
        streams[1]);
    // axpy is a*x + y, in this case c is a constant;
    axpy_c(n, alpha, tmp, extra); // pr = alpha * pr + extra;

    // Compute error norm;
    if (!error_on_cpu) {
        *final_error = euclidean_dist(n, pr, tmp, streams[2]);
    } else {
        cudaDeviceSynchronize();
        *final_error = euclidean_dist_cpu(n, pr, tmp);
    }

    // Store in "tmp" the updated values of PR;
    std::swap(tmp, pr);
    if (human_readable) {
        std::cout << "Error " << iter << ") " << *final_error << std::endl;
    }
    return *final_error <= tolerance;
}

////////////////////////////////
////////////////////////////////

template int pagerank<int, float>(
    int n,
    int e,
    int *csc_ptr,
    int *csc_ind,
    float *csc_val,
    float alpha,
    float tolerance,
    int max_iter,
    float *pagerank_vector,
    int *final_iteration,
    float *final_error,
    bool human_readable,
    bool error_on_cpu);

template int pagerank<int, double>(
    int n,
    int e,
    int *csc_ptr,
    int *csc_ind,
    double *csc_val,
    double alpha,
    float tolerance,
    int max_iter,
    double *pagerank_vector,
    int *final_iteration,
    double *final_error,
    bool human_readable,
    bool error_on_cpu);
