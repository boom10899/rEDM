#include "forecast_machine_cuda.h"

// __device__ void find_nearest_neighbors_cuda(const vec& dist, std::vector<size_t> &return_nearest_neighbors, size_t nn, std::vector<size_t> which_lib, double epsilon)
__device__ void find_nearest_neighbors_cuda_test()
{
    // printf("Test");
    return;
}

// __device__ void adjust_lib(const size_t curr_pred)
// {
//     // clear out lib indices we don't want from which_lib
//     if(exclusion_radius >= 0)
//     {
//         auto f = [&](const size_t curr_lib) {
//             return (curr_lib == curr_pred) || ((time[curr_lib] >= (time[curr_pred] - exclusion_radius)) && (time[curr_lib] <= (time[curr_pred] + exclusion_radius)));
//         };
//         which_lib.erase(std::remove_if(which_lib.begin(), which_lib.end(), f), which_lib.end());
//     }
//     else
//     {
//         which_lib.erase(std::remove(which_lib.begin(), which_lib.end(), curr_pred), which_lib.end());
//     }
//     return;
// }

__global__ void simplex_prediction_cuda(const size_t start, const size_t end, size_t *which_pred, size_t *which_lib)
{
    // int index = blockIdx.x * blockDim.x + threadIdx.x;
    // int stride = blockDim.x * gridDim.x;
    // // printf("Thread: %d, Block: %d\n", threadIdx.x, blockIdx.x);
    // find_nearest_neighbors_cuda_test();
    // for (int i = index; i < n; i += stride)
    //     y[i] = x[i] + y[i];

    printf("GPU\n");

    printf("Start: %lu\n", start);
    printf("End: %lu\n", end);

    for(size_t k = start; k < end; ++k)
    {
        size_t curr_pred = which_pred[k];

        printf("%lu\n", curr_pred);
        // printf("%lu\n", which_lib[k]);

        // find nearest neighbors
        // if(CROSS_VALIDATION)
        // {
            // size_t temp_lib = which_lib;
            // adjust_lib(curr_pred);
        //     // find_nearest_neighbors_cuda<<<1, 1>>>(distances[curr_pred], nearest_neighbors, nn, which_lib, epsilon);

        //     find_nearest_neighbors_cuda(distances[curr_pred], nearest_neighbors);
            // which_lib = temp_lib;
        // }
        // else
        // {

        //     find_nearest_neighbors(distances[curr_pred], nearest_neighbors);
        // }
    }
}

void call_cuda(const size_t start, const size_t end, std::vector<size_t> which_pred, std::vector<size_t> which_lib, size_t **which_lib_adjusted)
{
    size_t *which_pred_array;
    size_t *which_lib_array;
    size_t *which_lib_adjusted_array;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&which_pred_array, which_pred.size()*sizeof(size_t));
    cudaMallocManaged(&which_lib_array, which_lib.size()*sizeof(size_t));
    cudaMallocManaged(&which_lib_array, which_lib.size()*which_lib.size()*sizeof(size_t));

    std::copy(which_pred.begin(), which_pred.end(), which_pred_array);
    std::copy(which_lib.begin(), which_lib.end(), which_lib_array);

    // Run kernel on 1M elements on the GPU
    // int blockSize = 256;
    // int numBlocks = (N + blockSize - 1) / blockSize;
    simplex_prediction_cuda<<<1,1>>>(start, end, which_pred_array, which_lib_array);
    cudaError_t error = cudaGetLastError();    
    if (error != cudaSuccess)
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Free memory
    cudaFree(which_pred_array);
    cudaFree(which_lib_array);

    return;
}

std::vector<size_t> find_nearest_neighbors_cuda(const vec& dist, size_t nn, std::vector<size_t> which_lib, double epsilon) 
{
    if(nn < 1)
    {
        return sort_indices_cuda(dist, which_lib);
    }
    // else
    std::vector<size_t> neighbors;
    std::vector<size_t> nearest_neighbors;
    double curr_distance;

    if(nn > log(double(which_lib.size())))
    {
        // printf("True");
        neighbors = sort_indices_cuda(dist, which_lib);
        std::vector<size_t>::iterator curr_lib;

        // find nearest neighbors
        for(curr_lib = neighbors.begin(); curr_lib != neighbors.end(); ++curr_lib)
        {
            nearest_neighbors.push_back(*curr_lib);
            if(nearest_neighbors.size() >= nn)
                break;
        }
        if(curr_lib == neighbors.end())
            return nearest_neighbors;

        double tie_distance = dist[nearest_neighbors.back()];

        // check for ties
        for(++curr_lib; curr_lib != neighbors.end(); ++curr_lib)
        {
            if(dist[*curr_lib] > tie_distance) // distance is bigger
                break;
            nearest_neighbors.push_back(*curr_lib); // add to nearest neighbors
        }
    }
    else
    {
        // printf("False");
        size_t i;
        for(auto curr_lib: which_lib)
        {
            // distance to current neighbor under examination
            curr_distance = dist[curr_lib];
            
            // We want to include the current neighbor:
            //   if haven't populated neighbors vector, or
            //   if current neighbor is nearer than farthest away neighbor
            if(nearest_neighbors.size() < nn || 
               curr_distance <= dist[nearest_neighbors.back()])
            {
                // find the correct place to insert the current neighbor
                i = nearest_neighbors.size();
                while((i > 0) && (curr_distance < dist[nearest_neighbors[i-1]]))
                {
                    i--;
                }
                nearest_neighbors.insert(nearest_neighbors.begin()+i, curr_lib);

                // if we've added too many neighbors and there isn't a tie, then
                // pop off the farthest neighbor
                while((nearest_neighbors.size() > nn) &&
                   (dist[nearest_neighbors[nn-1]] < dist[nearest_neighbors.back()]))
                {
                    nearest_neighbors.pop_back();
                }
            }
        }
    }

    // filter for max_distance
    if(epsilon >= 0)
    {
        for(auto neighbor_iter = nearest_neighbors.begin(); neighbor_iter != nearest_neighbors.end(); ++neighbor_iter)
        {
            if(dist[*neighbor_iter] > epsilon)
            {
                nearest_neighbors.erase(neighbor_iter, nearest_neighbors.end());
                break;
            }
        }
    }

    return nearest_neighbors;
}

std::vector<size_t> sort_indices_cuda(const vec& v, std::vector<size_t> idx)
{
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    return idx;
}