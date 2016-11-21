#include <iostream>
#include <vector>
#include "profiler.h"
#include <algorithm>
#include <fstream>
#include <sstream>

const int block_size = 1024;

__global__ void scan( double *in_data, double *out_data, int n )
{
    __shared__ double temp[2 * block_size];

    int tid = threadIdx.x;
    int offset = 1;

    temp[tid] = in_data[tid];
    temp[tid + block_size] = in_data[tid + block_size];

    for( int d = n / 2; d > 0; d /= 2 )
    {
        __syncthreads ();

        if( tid < d )
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }

        offset *= 2;
    }

    if ( tid == 0 )
        temp[n-1] = 0;

    for ( int d = 1; d < n; d *= 2 )
    {
        offset /= 2;

        __syncthreads();

        if( tid < d )
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            double t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads ();

    out_data[2 * tid] = temp[2 * tid];
    out_data[2 * tid + 1] = temp[2 * tid + 1];
}

int main(int argc, char **argv)
{
    profiler prof;

    auto n = 2 * block_size;
    auto n_bytes = n * sizeof(double);
    std::vector<double> a(n);
    std::vector<double> b(n);
    std::vector<double> cpu_scan(n);

    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::mt19937 engine; 
    auto generator = std::bind(distribution, engine);
    std::generate_n(a.begin(), n, generator);

    prof.tic("cpu scan");
    cpu_scan[0] = 0.;
    for(int i = 1; i < n; ++i)
        cpu_scan[i] = cpu_scan[i - 1] + a[i - 1];
    prof.toc("cpu scan");

    double *in_dev = nullptr;
    double *out_dev = nullptr;

    cudaError_t cuerr = cudaMalloc( (void**)&in_dev, n_bytes);
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot allocate GPU memory for in_dev" << cudaGetErrorString(cuerr) << std::endl;
        return 1;
    }

    cuerr = cudaMalloc( (void**)&out_dev, n_bytes);
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot allocate GPU memory for out_dev" << cudaGetErrorString(cuerr) << std::endl;
        return 1;
    }

    cuerr = cudaMemcpy ( in_dev, a.data(), n_bytes, cudaMemcpyHostToDevice );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot copy data to device" << cudaGetErrorString(cuerr) << std::endl;
        return 1;
    }

    prof.tic("gpu scan");
    scan<<<1, block_size>>>(in_dev, out_dev, n);

    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot launch CUDA kernel 1 " << cudaGetErrorString(cuerr) << std::endl;
        return 1;
    }

    cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot synchronize CUDA kernel 1 " << cudaGetErrorString(cuerr) << std::endl;
        return 1;
    }
    
    cuerr = cudaMemcpy( b.data(), out_dev, n_bytes, cudaMemcpyDeviceToHost );
    
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot copy data from device " << cudaGetErrorString(cuerr) << std::endl;
        return 1;
    }
    prof.toc("gpu scan");

    for(auto i = 0; i < n; ++i)
        if( fabs(b[i] - cpu_scan[i]) > 1e-5 )
        {
            std::cout << "fail " << std::endl;
            std::cout << "a = " << a[i-1] << std::endl;
            std::cout << "b = " << b[i] << std::endl;
            std::cout << "cpu = " << cpu_scan[i] << std::endl;
            std::cout << "i = " << i << std::endl;
            cudaFree(in_dev);
            cudaFree(out_dev);
            return 1;
        }

    prof.report();

    cudaFree(in_dev);
    cudaFree(out_dev);

    return 0;
}
