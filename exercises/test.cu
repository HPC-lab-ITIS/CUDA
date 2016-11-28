#include <iostream>
#include <vector>
#include "profiler.h"
#include <algorithm>
#include <fstream>
#include <sstream>

const int block_size = 32 * 32;

__global__ void reduce_5(double* in_data, double* out_data)
{
    __shared__ double buf[block_size];

    int thread_rank = threadIdx.x + threadIdx.y * blockDim.x;
    int block_rank = blockIdx.x + blockIdx.y * gridDim.x;
    int idx = thread_rank + block_rank * blockDim.x * blockDim.y;

    buf[thread_rank] = in_data[idx];

    __syncthreads ();

    for (int s = block_size / 2; s > 0; s /= 2)
    {
        if ( thread_rank < s )
            buf[thread_rank] += buf[thread_rank + s];

        __syncthreads();
    }

    if( thread_rank == 0 )
        out_data[block_rank] = buf[thread_rank];
}


int main(int argc, char **argv)
{
    profiler prof;

    size_t n =  512 * 1024 * 1024;

    std::vector<double> a(n);

    std::uniform_real_distribution<double> distribution(1.0, 10.0);
    std::mt19937 engine; 
    auto generator = std::bind(distribution, engine);

    std::generate_n(a.begin(), n, generator);

    prof.tic("reduction cpu");
    double sum = std::accumulate(a.begin(), a.end(), 0.0);
    prof.toc("reduction cpu");

    double *a_dev[2];
    
    cudaError_t cuerr = cudaMalloc( (void**)&a_dev[0], n * sizeof(double));
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot allocate GPU memory for a_dev" << cudaGetErrorString(cuerr) << std::endl;
        return 1;
    }

    cuerr = cudaMalloc( (void**)&a_dev[1], n * sizeof(double));
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot allocate GPU memory for a_dev" << cudaGetErrorString(cuerr) << std::endl;
        return 1;
    }

    cuerr = cudaMemcpy ( a_dev[0], a.data(), n * sizeof(double), cudaMemcpyHostToDevice );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot copy data to device" << cudaGetErrorString(cuerr) << std::endl;
        return 1;
    }

    auto i = 0;
    size_t j = n;

    prof.tic("reduction gpu 5");
    for (j = n ; j >= block_size; j /= block_size, i ^= 1)
    {
        auto num_blocks = j / block_size;
        auto b_x = num_blocks;
        auto b_y = 1;
       
        if( num_blocks > 65536 )
        {
            b_x = block_size;
            b_y = num_blocks / b_x;
        }
        
        dim3 threads(32, 32, 1);
        dim3 blocks(b_x, b_y, 1);

        reduce_5<<<blocks, threads>>>(a_dev[i], a_dev[i^1]);

        cuerr = cudaGetLastError();
        if (cuerr != cudaSuccess)
        {
            std::cout << "Cannot launch CUDA kernel " << cudaGetErrorString(cuerr) << std::endl;
            return 1;
        }

        cuerr = cudaDeviceSynchronize();
        if (cuerr != cudaSuccess)
        {
            std::cout << "Cannot synchronize CUDA kernel " << cudaGetErrorString(cuerr) << std::endl;
            return 1;
        }
    }

    std::cout << "j = " << j << std::endl;
    std::vector<double> b(j);
    cuerr = cudaMemcpy( b.data(), a_dev[i], sizeof(double)*j, cudaMemcpyDeviceToHost );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot copy data from device " << cudaGetErrorString(cuerr) << std::endl;
        return 1;
    }

    double sum_dev = std::accumulate(b.begin(), b.end(), 0.0);

    prof.toc("reduction gpu 5");

    std::cout <<  "error = " <<  fabs( sum - sum_dev) / n  << std::endl;
    
    prof.report();

    cudaFree(a_dev[0]);
    cudaFree(a_dev[1]);

    return 0;
}
