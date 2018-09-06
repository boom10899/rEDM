#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <thread>
#include <stdexcept>
#include <algorithm>
#include <math.h>
#include "data_types.h"
#include <typeinfo>

#include <thrust/sort.h>
#include <thrust/functional.h>

std::vector<size_t> sort_indices_cuda(const vec& v, std::vector<size_t> idx);
