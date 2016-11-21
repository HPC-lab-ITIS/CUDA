#include <iostream>
#include <vector>
#include "profiler.h"
#include <algorithm>

const int block_x = 16;
const int block_y = 8;

__global__ void transpose_naive(double *a, const double *b)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y *  blockDim.y + threadIdx.y;

  a[y + x*blockDim.x*gridDim.x] = b[x + y*blockDim.x*gridDim.x];
}

__global__ void transpose_coalesced(double *a, const double *b)
{

    __shared__ double tile[block_x][block_x];

    int x = blockIdx.x * block_x + threadIdx.x;
    int y = blockIdx.y * block_x + threadIdx.y;
    int width = gridDim.x * block_x;

    for (int j = 0; j < block_x; j += block_y)
        tile[threadIdx.y+j][threadIdx.x] = b[(y+j)*width + x];

    __syncthreads();

    x = blockIdx.x * block_x + threadIdx.x;
    y = blockIdx.y * block_x + threadIdx.y;

    for (int j = 0; j < block_x; j += block_y)
        a[(y + j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

__global__ void transpose_final(double *a, const double *b)
{
    __shared__ double tile[block_x][block_x + 1];

    int x = blockIdx.x * block_x + threadIdx.x;
    int y = blockIdx.y * block_x + threadIdx.y;
    int width = gridDim.x * block_x;

    for (int j = 0; j < block_x; j += block_y)
        tile[threadIdx.y+j][threadIdx.x] = b[(y+j)*width + x];

    __syncthreads();

    x = blockIdx.x * block_x + threadIdx.x;
    y = blockIdx.y * block_x + threadIdx.y;

    for (int j = 0; j < block_x; j += block_y)
        a[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

int gpu_trans(std::vector<double> &a, std::vector<double> &b, size_t n)
{
    auto n_bytes = n * n * sizeof(double);
    double *a_dev = nullptr, *b_dev = nullptr;
    profiler prof;

    cudaError_t cuerr = cudaMalloc ( (void**)&a_dev, n_bytes );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot allocate GPU memory for a_dev" << cudaGetErrorString(cuerr);
        return 1;
    }

    cuerr = cudaMalloc ( (void**)&b_dev, n_bytes );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot allocate GPU memory for b_dev" << cudaGetErrorString(cuerr);
        return 1;
    }

    cuerr = cudaMemcpy ( a_dev, a.data(), n_bytes, cudaMemcpyHostToDevice );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot copy data from a to a_dev" << cudaGetErrorString(cuerr);
        return 1;
    }

    dim3 block_size(block_x, block_y);
    dim3 grid_size(n/block_x, n/block_y);
    dim3 grid_size_1(n/block_x, n/block_x);

    cudaThreadSynchronize();
    prof.tic("Naive transpose");
    for(auto i = 0; i < 100; ++i)
        transpose_naive<<<grid_size, block_size>>>(a_dev, b_dev);
    cudaThreadSynchronize();
    prof.toc("Naive transpose");

    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot launch CUDA kernel " << cudaGetErrorString(cuerr);
        return 1;
    }

    cudaThreadSynchronize();
    prof.tic("Coalesced transpose");
    for(auto i = 0; i < 100; ++i)
        transpose_coalesced<<<grid_size_1, block_size>>>(a_dev, b_dev);
    cudaThreadSynchronize();
    prof.toc("Coalesced transpose");

    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot launch CUDA kernel " << cudaGetErrorString(cuerr);
        return 1;
    }

    cudaThreadSynchronize();
    prof.tic("No bank conflicts transpose");
    for(auto i = 0; i < 100; ++i)
        transpose_final<<<grid_size_1, block_size>>>(a_dev, b_dev);
    cudaThreadSynchronize();
    prof.toc("No bank conflicts transpose");

    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot launch CUDA kernel " << cudaGetErrorString(cuerr);
        return 1;
    }

    cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot synchronize CUDA kernel " << cudaGetErrorString(cuerr);
        return 1;
    }

    cuerr = cudaMemcpy ( b.data(), b_dev, n_bytes, cudaMemcpyDeviceToHost );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot copy data from c_dev to c " << cudaGetErrorString(cuerr);
        return 1;
    }

    cudaFree(a_dev);
    cudaFree(b_dev);

    prof.report();

    return 0;
}

int main()
{
    const size_t n = 13*1024;
    std::vector<double> a(n*n, 0.);
    std::vector<double> b(n*n, 0.);
    std::vector<double> b_host(n*n, 0.);
    profiler prof;

    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::mt19937 engine; // Mersenne twister MT19937
    auto generator = std::bind(distribution, engine);
    std::generate_n(a.begin(), n*n, generator);

    gpu_trans(a, b, n);

    prof.tic("Sequential transpose");
    for(auto i = 0; i < n; ++i)
        for(auto j = 0; j < n; ++j)
            if(i!=j)
                b[j + i*n] = a[i + j*n];
    prof.toc("Sequential transpose");

    prof.report();

    return 0;
}
