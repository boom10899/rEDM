#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <thread>
#include <stdexcept>
#include <algorithm>
#include <math.h>
#include "data_types.h"

void call_cuda(const size_t start, const size_t end, std::vector<size_t> which_pred, std::vector<size_t> which_lib, size_t **which_lib_adjusted);
std::vector<size_t> find_nearest_neighbors_cuda(const vec& dist, size_t nn, std::vector<size_t> which_lib, double epsilon);
std::vector<size_t> sort_indices_cuda(const vec& v, std::vector<size_t> idx);
