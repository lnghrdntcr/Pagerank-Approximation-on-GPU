#include <chrono>
#include <getopt.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <map>
#include <algorithm>

#include "pagerank.cuh"
#include "pagerank_pipeline.cuh"
#include "sample_graph.hpp"
#include "utils.hpp"

#define DEBUG false
#define num_type double 
////////////////////////////////
////////////////////////////////

using std::cout;
using std::endl;
using std::string;
using std::vector;

int main(int argc, char *argv[]) {
    // Input variables, with default values;
    string default_graph_type = "full_undirected";
    string csc_data_file = "/home/fra/University/HPPS/Approximate-PR/new_ds/scf/val.txt";
    string csc_ptr_file= "/home/fra/University/HPPS/Approximate-PR/new_ds/scf/non_zero.txt";
    string csc_indices_file  = "/home/fra/University/HPPS/Approximate-PR/new_ds/scf/col_idx.txt";

    int max_iter = 200;
    num_type min_norm_error = 0.0;
    num_type dangling_factor = 0.85f;

    bool use_cpu_gpu_async = false;
    bool use_sample_graph = false;
    bool human_readable = false;
    bool error_on_cpu = false;

    // Vectors where data are stored;
    vector<int> g_ind_vec;
    vector<int> g_ptr_vec;
    vector<double> g_data_vec;

    /* 
    * `w`, `i`, `p`: paths to the input graph, stored as CSC. Input parameters correspond to the *data* file, the *indices* file, and the *pointer* file.
    * `a`: if present, use the `pipeline` version of PageRank.
    * `m`: maximum number of iterations.
    * `e`: minimum error.
    * `d`: dangling factor.
    * `s`: use a small example graph instead of the input files.
    * `h`: if present, print all debug information, else a single summary line at the end
    * `c`: if present, compute the error term on CPU, synchronously. If both 'c' and 'a' are present, 'a' is used.
    */
    int opt;
    static struct option long_options[] =
        {
            {"human_readable", no_argument, 0, 'h'},
            {"use_cpu_gpu_async", no_argument, 0, 'a'},
            {"error_on_cpu", no_argument, 0, 'c'},
            {"use_sample_graph", no_argument, 0, 's'},
            {"csc_data_file", required_argument, 0, 'w'},
            {"csc_indices_file", required_argument, 0, 'i'},
            {"csc_ptr_file", required_argument, 0, 'p'},
            {"max_iter", required_argument, 0, 'm'},
            {"min_norm_error", required_argument, 0, 'e'},
            {"dangling_factor", required_argument, 0, 'd'},
            {0, 0, 0, 0}};
    // getopt_long stores the option index here;
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, "hacw:i:p:m:e:sd:", long_options, &option_index)) != EOF) {
        switch (opt) {
            case 'h':
                human_readable = true;
                break;
            case 'a':
                use_cpu_gpu_async = true;
                break;
            case 'c':
                error_on_cpu = true;
                break;
            case 'w':
                csc_data_file = optarg;
                break;
            case 'i':
                csc_indices_file = optarg;
                break;
            case 'p':
                csc_ptr_file = optarg;
                break;
            case 'm':
                max_iter = atoi(optarg);
                break;
            case 'e':
                min_norm_error = atof(optarg);
                break;
            case 's':
                use_sample_graph = true;
                break;
            case 'd':
                dangling_factor = atof(optarg);
                break;
            default:
                return 0;
        }
    }

    if (human_readable) {
        cout << "------------------------------\n"
             << "- Pagerank GPU Test          -\n"
             << "------------------------------" << endl;
    }

    if (!use_sample_graph) {
        if (human_readable) {
            cout << "Using files:" << endl;
            cout << "- " << csc_indices_file << endl;
            cout << "- " << csc_ptr_file << endl;
            cout << "- " << csc_data_file << endl;
        }

        int res = readArrayFile(csc_indices_file, &g_ind_vec);
        if (res) {
            if (human_readable) {
                cout << "error reading indices file" << endl;
            }
            return 1;
        }
        res = readArrayFile(csc_ptr_file, &g_ptr_vec);
        if (res) {
            if (human_readable) {
                cout << "error reading pointer file" << endl;
            }
            return 1;
        }
        res = readArrayFile(csc_data_file, &g_data_vec);
        if (res) {
            if (human_readable) {
                cout << "error reading data file" << endl;
            }
            return 1;
        }
        // Override values in data file;
        for (int i = 0; i < g_data_vec.size(); i++) {
            g_data_vec[i] = 1.0f;
        }
    } else {
        if (human_readable) {
            cout << "Using sample graph" << endl;
        }
        // Use sample graph;
        g_ind_vec.assign(g_indices, g_indices + E);
        g_ptr_vec.assign(g_ptr, g_ptr + V + 1);
        g_data_vec.assign(g_data, g_data + E);
    }

    // Print settings;
    if (human_readable) {
        cout << "------------------------------" << endl;
        cout << "- Dangling factor: " << dangling_factor << endl;
        cout << "- Max. Iteration: " << max_iter << endl;
        cout << "- Min. Error: " << min_norm_error << endl;
        if (use_cpu_gpu_async) {
            cout << "- Using CPU+GPU Async PR" << endl;
        } else if (error_on_cpu) {
            cout << "- Using CPU+GPU Sync PR" << endl;
        } else {
            cout << "- Using full GPU PR" << endl;
        }
        cout << "------------------------------" << endl;
    }

    // Number of vertices;
    int n = (int)g_ptr_vec.size() - 1;
    // Number of edges;
    int e = (int)g_ind_vec.size();

    if (human_readable) {
        cout << "------------------------------" << endl;
        cout << "- V: " << n << endl;
        cout << "- E: " << e << endl;
        cout << "------------------------------" << endl;
    }

    num_type *pr = (num_type *)calloc(n, sizeof(num_type));

    int final_iteration = 0;
    num_type final_error = 0.0f;

    // Pass vectors as arrays, and use the right version of PR;
    pagerank(n, e, g_ptr_vec.data(), g_ind_vec.data(), g_data_vec.data(),
             dangling_factor, min_norm_error, max_iter, pr,
             &final_iteration, &final_error, human_readable, error_on_cpu);

    std::map<int, num_type> pr_map;
    std::vector<std::pair<int, num_type>> sorted_pr;
    std::vector<int> sorted_pr_idxs;

    for (int i = 0; i < n; ++i) {
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
//        std::cout << pair.first << std::endl;
    }

    if (DEBUG) {
        std::cout << "Checking results..." << std::endl;

        std::ifstream results;
        results.open("/home/fra/University/HPPS/Approximate-PR/new_ds/scf/results.txt");

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

        std::cout << "Percentage of error: " << (((double) errors_real) / (n)) * 100 << "%\n" << std::endl;
    }



}
