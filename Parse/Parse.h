//
// Created by fra on 19/04/19.
//

#ifndef APPROXIMATE_PR_PARSE_H
#define APPROXIMATE_PR_PARSE_H

#endif //APPROXIMATE_PR_PARSE_H

#include <iostream>
#include <vector>

typedef struct csc_t {
    std::vector<double> val;
    std::vector<int> non_zero;
    std::vector<int> col_idx;
} csc_t;

csc_t parse_dir(std::string dir_path);