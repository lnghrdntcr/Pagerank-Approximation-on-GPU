//
// Created by fra on 26/04/19.
//

#include "Utils.h"

template<typename T>
bool check_error(T *e, const T error, const unsigned DIMV) {
    for (int i = 0; i < DIMV; ++i) {
        if (e[i] > error) return false;
    }
    return true;
}

