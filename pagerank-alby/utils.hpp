#pragma once

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>

inline void random_ints(int *v, int n, int max = RAND_MAX) {
    for (int i = 0; i < n; i++) {
        v[i] = rand() % max;
    }
}

template <typename T>
inline void print_array(T *v, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << v[i];
        if (i < n - 1) {
            std::cout << " ";
        }
    }
    std::cout << std::endl;
}

template <typename T>
inline void print_array_indexed(T *v, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << i << ") " << v[i] << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
inline int readArrayFile(std::string filename, std::vector<T> *mem_buffer) {
    std::ifstream inputFile(filename.c_str());
    if (inputFile.good()) {
        // Push items into a vector
        T current_value = 0;
        int i = 0;
        while (inputFile >> current_value) {
            (*mem_buffer).push_back(current_value);
            i++;
        }
        // Close the file.
        inputFile.close();
        return 0;
    } else {
        std::cout << "Error reading file!" << std::endl;
        return 1;
    }
}
