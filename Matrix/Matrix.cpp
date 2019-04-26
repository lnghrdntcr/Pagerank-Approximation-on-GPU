//
// Created by Francesco Sgherzi on 26/04/19.
//

#include <set>

#include "Matrix.h"


template<typename T>
void generate_sparse_matrix(T *matrix, const unsigned int DIMV, const unsigned int min_sparse) {

    // for all rows
    for (int i = 0; i < DIMV; ++i) {

        int num_zeroes = rand() % (DIMV - min_sparse) + min_sparse;
        std::set<int> zero_idxs;

        zero_idxs.insert(i);
        for (int j = 0; j < num_zeroes; ++j) {
            int r_idx = rand() % DIMV;
            zero_idxs.insert(r_idx);
        }

        for (int j = 0; j < DIMV; ++j) {
            if (zero_idxs.find(j) == zero_idxs.end() && (DIMV - zero_idxs.size()) != 0) {
                matrix[i * DIMV + j] = (T) 1.0 / (DIMV - zero_idxs.size());
            }
        }

    }
}

template<typename T>
void fill_spm(T *matrix, const unsigned int DIMV) {
    for (int i = 0; i < DIMV; ++i) {
        int count_zero = 0;
        for (int j = 0; j < DIMV; ++j) {
            if (matrix[i * DIMV + j] == 0.0) count_zero++;
        }
        if (count_zero == DIMV) matrix[i * DIMV + i] = 1;
    }
}

template<typename T>
void transpose(T *out, T *in, const unsigned DIMV) {

    for (int i = 0; i < DIMV; ++i) {
        for (int j = 0; j < DIMV; ++j) {
            out[i * DIMV + j] = in[j * DIMV + i];
        }
    }

}

template<typename T>
void to_csc(T *csc_val, int *csc_non_zero, int *csc_col_idx, T *src, const unsigned DIMV, const unsigned non_zero) {

    unsigned val_idx = 0;

    csc_non_zero[0] = 0;

    for (int i = 0; i < DIMV; ++i) {

        csc_non_zero[i + 1] = csc_non_zero[i];

        for (int j = 0; j < DIMV; ++j) {

            if (src[i * DIMV + j] > 0) {
                csc_val[val_idx] = src[i * DIMV + j];
                csc_non_zero[i + 1]++;
                csc_col_idx[val_idx] = j;

                val_idx++;
            }

        }

    }
}

template<typename T>
unsigned int count_non_zero(T *m, const unsigned int DIMV) {
    int sum = 0;

    for (int i = 0; i < DIMV * DIMV; ++i) {
        if (m[i] > 0) sum++;
    }

    return sum;
}