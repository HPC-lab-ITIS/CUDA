#include <iostream>
#include <vector>
#include "profiler.h"
#include <algorithm>

__global__ void reduce_1(double* in, int n, int m)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    for(int step = 1; step < blockDim.x; step *= 2)
    {
        if( ( threadIdx.x % (2*step) == 0) && (step + threadIdx.x < blockDim.x ) )
            in[n + i] += in[n + i + step];

        __syncthreads();
    }

    if(threadIdx.x == 0)
        in[m + blockIdx.x] = in[n + blockIdx.x*blockDim.x];
}


int main()
{
    const auto n = 1024*1024*33;
    double *a = nullptr;
    profiler prof;

    cudaError_t cuerr = cudaMallocManaged(&a, 2 * n * sizeof(double));

    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot allocate GPU memory for a_dev" << cudaGetErrorString(cuerr);
        return 1;
    }

    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::mt19937 engine;
    auto generator = std::bind(distribution, engine);
    std::generate_n(a, n, generator);

    auto threads = 1024;
    auto blocks = n / threads;

    double sum = 0.;

    prof.tic("reduction");
    for(int i = 0; i < n; ++i)
        sum += a[i];
    prof.toc("reduction");

    auto i1 = 0;
    auto i2 = n;
    auto l_b = 0;
    prof.tic("reduction gpu 1");
    cudaThreadSynchronize();
    for(auto i = blocks; i != 0; i /= threads)
    {
        reduce_1<<<i, threads>>>(a, i1, i2);
        cuerr = cudaGetLastError();
        if (cuerr != cudaSuccess)
        {
            std::cout << "Cannot launch CUDA kernel " << cudaGetErrorString(cuerr);
            return 1;
        }
        std::swap(i1, i2);
        l_b = i;
    }

    if(l_b < threads)
        reduce_1<<<1, l_b>>>(a, i1, i2);

    cudaThreadSynchronize();
    prof.toc("reduction gpu 1");


    if( fabs( sum - a[i2] ) > 1e-5 )
    {
        std::cout << "fail" << std::endl;
        std::cout << a[i1] << std::endl;
        std::cout << a[i2] << std::endl;
        std::cout << sum << std::endl;
        cudaFree(a);
        return 1;
    }

    prof.report();

    cudaFree(a);

    return 0;
}
