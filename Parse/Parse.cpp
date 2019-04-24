//
// Created by Francesco Sgherzi on 19/04/19.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "Parse.h"

csc_t parse_dir (const std::string dir_path){

    std::cout << "Parsing " << dir_path << std::endl;

    std::stringstream ss_val;
    std::stringstream ss_n_zero;
    std::stringstream ss_col_idx;

    std::ifstream val, non_zero, col_idx;

    std::vector<double> csc_val;
    std::vector<int> csc_non_zero;
    std::vector<int> csc_col_idx;

    csc_t csc;

    ss_val << dir_path << "/" << "val.txt";
    ss_n_zero << dir_path << "/" << "non_zero.txt";
    ss_col_idx << dir_path << "/" << "col_idx.txt";

    val.open(ss_val.str());
    non_zero.open(ss_n_zero.str());
    col_idx.open(ss_col_idx.str());

    if (!val || !non_zero || !col_idx) {
        std::cerr << "Error reading file" << std::endl;
        exit(1);
    }


    int tmp2;
    double tmp1;

    while (val >> tmp1) {
        csc_val.push_back(tmp1);
    }

    while (non_zero >> tmp2) {
        csc_non_zero.push_back(tmp2);
    }

    while (col_idx >> tmp2) {
        csc_col_idx.push_back(tmp2);
    }

    csc.val = csc_val;
    csc.non_zero = csc_non_zero;
    csc.col_idx = csc_col_idx;

    return csc;

}