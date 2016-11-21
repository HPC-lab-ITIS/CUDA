#include <iostream>
#include <vector>
#include "profiler.h"
#include <algorithm>
#include <fstream>
#include <sstream>

const int block_size = 1024;

__global__ void reduce_5(double* in_data, double* out_data)
{
    __shared__ double buf[block_size];

    int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    buf[threadIdx.x] = in_data[i] + in_data[i + blockDim.x];

    __syncthreads ();

    for (int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if ( threadIdx.x < s )
            buf[threadIdx.x] += buf[threadIdx.x + s];

        __syncthreads();
    }

    if( threadIdx.x == 0 )
        out_data[blockIdx.x] = buf[threadIdx.x];
}


int main(int argc, char **argv)
{
    profiler prof;
    auto n_par = 32;

    if (argc>1)
    {
        std::istringstream iss(argv[1]);
        iss >> n_par;
    }

    int n = n_par * block_size * block_size;

    std::vector<double> a(n);

    std::uniform_real_distribution<double> distribution(0.0, 1.0);
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

    int i = 0;
    int j;

    prof.tic("reduction gpu 5");
    for (j = n ; j >= block_size; j /= (2*block_size), i ^= 1)
    {

        dim3 threads(block_size, 1, 1);
        dim3 blocks(j / (2*block_size), 1, 1);

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

    std::vector<double> b(j);
    cuerr = cudaMemcpy( b.data(), a_dev[i], sizeof(double)*j, cudaMemcpyDeviceToHost );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot copy data from device " << cudaGetErrorString(cuerr) << std::endl;
        return 1;
    }

    double sum_dev = std::accumulate(b.begin(), b.end(), 0.0);

    prof.toc("reduction gpu 5");

    if( fabs( sum - sum_dev) > 1e-5 )
    {
        std::cout <<  sum  << std::endl;
        std::cout <<  sum_dev  << std::endl;
        cudaFree(a_dev[0]);
        cudaFree(a_dev[1]);
        return 1;
    }

    prof.report();

    cudaFree(a_dev[0]);
    cudaFree(a_dev[1]);

    return 0;
}
