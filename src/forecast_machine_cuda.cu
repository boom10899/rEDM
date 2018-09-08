#include "forecast_machine_cuda.h"

struct sortIndices
{
    double *v;
    sortIndices(double * _v): v(_v) {};

    __host__ __device__
    bool operator()(size_t i1, size_t i2) const 
    {
        return v[i1] < v[i2];
    }
};

std::vector<size_t> sort_indices_cuda(const vec& v, std::vector<size_t> idx)
{
    thrust::device_vector<size_t> d_idx = idx;
    thrust::device_vector<double> d_v = v;

    // for(int i = 0; i < d_v.size(); i++)
    //     std::cout << "D[" << i << "] = " << d_v[i] << std::endl;
    // std::cout << "Size: idx = " << idx.size() << " | v = " << v.size() << std::endl;

    thrust::sort(d_idx.begin(), d_idx.end(), sortIndices(thrust::raw_pointer_cast(d_v.data())));
    thrust::copy(d_idx.begin(), d_idx.end(), idx.begin());

    return idx;
}