//
// Created by fra on 19/04/19.
//

#ifndef APPROXIMATE_PR_PARSE_H
#define APPROXIMATE_PR_PARSE_H

#endif //APPROXIMATE_PR_PARSE_H

#include <iostream>
#include <vector>
#define num_type double
#define fixed_num_type long long unsigned

typedef struct csc_t {
    std::vector<num_type> val;
    std::vector<int> non_zero;
    std::vector<int> col_idx;
} csc_t;

typedef struct csc_fixed_t {
    std::vector<fixed_num_type> val;
    std::vector<int> non_zero;
    std::vector<int> col_idx;
} csc_fixed_t;

csc_t parse_dir(std::string dir_path, bool debug);