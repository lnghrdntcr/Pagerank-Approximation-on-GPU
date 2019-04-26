//
// Created by fra on 26/04/19.
//

#ifndef APPROXIMATE_PR_MATRIX_H
#define APPROXIMATE_PR_MATRIX_H

template<typename T>
void generate_sparse_matrix(T *matrix, unsigned DIMV, unsigned min_sparse);


template<typename T>
void fill_spm(T *matrix, unsigned DIMV);

template<typename T>
void transpose(T *out, T *in, unsigned DIMV);

template<typename T>
void to_csc(T *csc_val, int *csc_non_zero, int *csc_col_idx, T *src, unsigned DIMV, unsigned non_zero);

template<typename T>
unsigned int count_non_zero(T *m, unsigned DIMV);

#endif //APPROXIMATE_PR_MATRIX_H
