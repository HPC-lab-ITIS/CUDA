#include <iostream>
#include <vector>
#include "profiler.h"
#include <algorithm>
#include <fstream>
#include <sstream>

__global__ void reduce_4(double* in, int n, int m)
{
    __shared__ double buf[1024];
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    buf[threadIdx.x] = in[n + i];

    __syncthreads();

    int size = blockDim.x;

    for(int step = blockDim.x / 2; step > 0; step /= 2)
    {
        if( ( threadIdx.x < step ) )
            buf[threadIdx.x] += buf[threadIdx.x + step];

        if( ( size % 2 != 0) && ( threadIdx.x == 0 ) )
            buf[threadIdx.x] += buf[2*step];

        __syncthreads();

        size = step;
    }

    if(threadIdx.x == 0)
        in[m + blockIdx.x] = buf[0];
}


int main(int argc, char **argv)
{
    auto n_par = 32;

    if (argc>1)
    {
        std::istringstream iss(argv[1]);
        iss >> n_par;
    }


    const auto n = 1024 * 1024 * n_par;
    std::vector<double> a(n);
    double *a_dev = nullptr;
    profiler prof;

    cudaError_t cuerr = cudaMalloc( (void**)&a_dev, 2 * n * sizeof(double));

    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot allocate GPU memory for a_dev" << cudaGetErrorString(cuerr);
        return 1;
    }

    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::mt19937 engine; 
    auto generator = std::bind(distribution, engine);
    std::generate_n(a.begin(), n, generator);

    cuerr = cudaMemcpy ( a_dev, a.data(), n * sizeof(double), cudaMemcpyHostToDevice );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot copy data from a to a_dev" << cudaGetErrorString(cuerr);
        return 1;
    }

    double sum = 0.;
    prof.tic("reduction cpu");
    sum = std::accumulate(a.begin(), a.end(), 0.0);
    prof.toc("reduction cpu");

    auto threads = 1024;
    auto blocks = n / threads;
    auto i1 = 0;
    auto i2 = n;
    auto l_b = 0;
    prof.tic("reduction gpu 4");
    cudaThreadSynchronize();
    for(auto i = blocks; i > 1; i /= threads)
    {
        reduce_4<<<i, threads>>>(a_dev, i1, i2);
        cuerr = cudaGetLastError();
        if (cuerr != cudaSuccess)
        {
            std::cout << "Cannot launch CUDA kernel " << cudaGetErrorString(cuerr);
            return 1;
        }
        std::swap(i1, i2);
        l_b = i;
    }

    reduce_4<<<1, l_b>>>(a_dev, i1, i2);
    cudaThreadSynchronize();
    prof.toc("reduction gpu 4");

    cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot synchronize CUDA kernel " << cudaGetErrorString(cuerr);
        return 1;
    }

    double sum_dev = 0;
    cuerr = cudaMemcpy ( &sum_dev, a_dev + i2, sizeof(double), cudaMemcpyDeviceToHost );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot copy data from device " << cudaGetErrorString(cuerr);
        return 1;
    }

    if( fabs( sum - sum_dev ) > 1e-5 )
    {
        std::cout <<  fabs( sum - sum_dev ) << std::endl;
        cudaFree(a_dev);
        return 1;
    }

    prof.report();

    cudaFree(a_dev);

    return 0;
}
