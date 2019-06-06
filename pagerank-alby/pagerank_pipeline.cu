#include <atomic>
#include <chrono>
#include <cub/cub.cuh>
#include <stdio.h>
#include <stdlib.h>
#include <thread>

#include "graph_utils.cuh"
#include "pagerank_pipeline.cuh"
#include "utils.hpp"

////////////////////////////////
////////////////////////////////

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

std::atomic_size_t stage;   // Current iteration, sed to synchronize CPU and GPU;
std::atomic_bool converged; // True if PR has converged;

template <typename IndexType, typename ValueType>
int pagerank_pipeline(
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
    float *final_error,
    bool human_readable) {

    auto start_pr = clock_type::now();

    int max_it = max_iter > 0 ? max_iter : 100;                                // Maximum number of iterations;
    float tol = (tolerance < 1.0f && tolerance >= 0.0f) ? tolerance : 1.0E-6f; // Tolerance;
    ValueType random_probability = static_cast<ValueType>(1.0 / n);            // Initial values of PR;

    stage.store(1);
    converged.store(false);

    if (alpha <= 0.0f || alpha >= 1.0f) {
        return -1;
    }

    bool *dangling_bitmap = 0; // Bitmap with values = 1 for dangling vertices;
    ValueType *tmp = 0;        // Temporary PageRank values;
    ValueType *pr_mng = 0;

    // Temporary storage allocated on GPU;
    void *cub_d_temp_storage = NULL;
    size_t cub_temp_storage_bytes = 0;

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
    memcpy(csc_ptr_mng, csc_ptr, sizeof(IndexType) * (n + 1));
    memcpy(csc_ind_mng, csc_ind, sizeof(IndexType) * e);
    memcpy(csc_val_mng, csc_val, sizeof(ValueType) * e);

    cudaCheckError();

    auto start_pr_preprocessing = clock_type::now();

    // Initialize PageRank values with 1/N;
    fill(n, pr_mng, random_probability);
    fill(n, tmp, random_probability);

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

    // "Fake" matrix multiplication, used to compute the size of cub_d_temp_storage and allocate it.
    // If cub_d_temp_storage is NULL no matrix multiplication is done, but it is used only for initialization.
    // Size of temporary storage is saved in cub_temp_storage_bytes;
    cub::DeviceSpmv::CsrMV(cub_d_temp_storage, cub_temp_storage_bytes, csc_val_mng,
                           csc_ptr_mng, csc_ind_mng, tmp, pr_mng, n, n, e);
    // Allocate temporary storage on GPU;
    cudaMallocManaged(&cub_d_temp_storage, cub_temp_storage_bytes);
    cudaCheckError();

    auto end_pr_preprocessing = clock_type::now();
    auto start_pr_main = clock_type::now();

    // Main PR iterations;
    if (human_readable) {
        std::cout << "- Starting PR Computation ----" << std::endl;
    }

    std::thread gpu_thread(
        gpu_controller<IndexType, ValueType>,
        n,
        e,
        csc_ptr_mng,
        csc_ind_mng,
        csc_val_mng,
        alpha,
        dangling_bitmap,
        max_it,
        std::ref(tmp),
        cub_d_temp_storage,
        cub_temp_storage_bytes,
        std::ref(pr_mng),
        final_iteration,
        human_readable);
    std::thread cpu_thread(cpu_controller<ValueType>, std::ref(tmp), std::ref(pr_mng), tol, final_error, n, max_it, human_readable);

    gpu_thread.join();
    cpu_thread.join();

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
        std::cout << n << ", " << e << ", "
                  << "cpu+gpu_pipeline"
                  << ", " << *final_iteration << ", "
                  << *final_error << ", " << duration << ", " << duration_preproc << ", " << duration_main << std::endl;
    }

    return converged ? 0 : 1;
}

////////////////////////////////
////////////////////////////////

template <typename ValueType>
void cpu_controller(ValueType *&tmp, ValueType *&pr, float tolerance, float *final_error, int n, int max_it, bool human_readable) {
    auto tot_time = 0;
    size_t current_it = 1;
    while (!converged.load() && current_it <= max_it) {

        if (current_it < stage.load()) {
            cudaDeviceSynchronize();
            auto start = clock_type::now();
            *final_error = euclidean_dist_cpu(n, pr, tmp);
            auto end = clock_type::now();
            tot_time += chrono::duration_cast<chrono::milliseconds>(end - start).count();
            converged.store(*final_error < tolerance);
            if (human_readable) {
                std::cout << "Error " << current_it << ") " << *final_error << std::endl;
            }
            current_it++;

            // Store in "tmp" the updated values of PR, if we need to do more iterations;
            if (current_it <= max_it && !converged.load()) {
                std::swap(tmp, pr);
            }
        }
    }
    if (human_readable) {
        std::cout << "Exec. Time CPU: " << tot_time << " ms" << std::endl;
    }
}

////////////////////////////////
////////////////////////////////

template <typename IndexType, typename ValueType>
void gpu_controller(
    IndexType n,
    IndexType e,
    IndexType *csc_ptr,
    IndexType *csc_ind,
    ValueType *csc_val,
    ValueType alpha,
    bool *dangling_bitmap,
    int max_it,
    ValueType *&tmp,
    void *cub_d_temp_storage,
    size_t cub_temp_storage_bytes,
    ValueType *&pr,
    int *final_iteration,
    bool human_readable) {
    auto tot_time = 0;
    size_t current_it = 1; // Could also use stage.load() here, but using a private var is faster
    while (!converged.load() && current_it <= max_it) {
        *final_iteration = current_it;
        auto start = clock_type::now();
        pagerank_iteration(
            n,
            e,
            csc_ptr,
            csc_ind,
            csc_val,
            alpha,
            dangling_bitmap,
            tmp,
            cub_d_temp_storage,
            cub_temp_storage_bytes,
            pr);
        auto end = clock_type::now();
        tot_time += chrono::duration_cast<chrono::milliseconds>(end - start).count();
        current_it++;
        stage.fetch_add(1);
    }
    if (human_readable) {
        std::cout << "Exec. Time GPU: " << tot_time << " ms" << std::endl;
    }
}

////////////////////////////////
////////////////////////////////

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
) {
    // Dangling factor, multiply current PR by dangling bitmap;
    ValueType dot_res = dot(n, dangling_bitmap, tmp);
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
        tmp, // x vector, it contains the current PR values;
        pr,  // y vector, where the new PR values are stored;
        n,   // Number of rows of A;
        n,   // Number of columns of A;
        e    // Non-zero elements in A, i.e. |E|
    );
    // cudaDeviceSynchronize();

    // axpy is a*x + y, in this case c is a constant;
    axpy_c(n, alpha, pr, extra); // pr = alpha * pr + extra;
    // cudaDeviceSynchronize();
}

////////////////////////////////
////////////////////////////////

template int pagerank_pipeline<int, float>(
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
    bool human_readable);

template int pagerank_pipeline<int, double>(
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
    float *final_error,
    bool human_readable);