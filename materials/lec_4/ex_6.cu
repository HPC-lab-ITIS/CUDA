#include <iostream>
#include <vector>
#include "profiler.h"
#include <algorithm>
#include <cublas_v2.h>


int main()
{
    const auto n = 208*32;
    std::vector<double> a(n*n,0.);
    std::vector<double> b(n*n,0.);
    std::vector<double> c(n*n,0.);
    std::vector<double> c_host(n*n,0.);
    profiler prof;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::mt19937 engine; // Mersenne twister MT19937
    auto generator = std::bind(distribution, engine);
    std::generate_n(a.begin(), n, generator);
    std::generate_n(b.begin(), n, generator);
    auto n_bytes = n * n * sizeof(double);
    double *a_dev = nullptr, *b_dev = nullptr, *c_dev = nullptr;

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

    cuerr = cudaMalloc ( (void**)&c_dev, n_bytes );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot allocate GPU memory for c_dev" << cudaGetErrorString(cuerr);
        return 1;
    }

    cuerr = cudaMemcpy ( a_dev, a.data(), n_bytes, cudaMemcpyHostToDevice );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot copy data from a to a_dev" << cudaGetErrorString(cuerr);
        return 1;
    }

    cuerr = cudaMemcpy ( b_dev, b.data(), n_bytes, cudaMemcpyHostToDevice );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot copy data from a to b_dev" << cudaGetErrorString(cuerr);
        return 1;
    }
   
    cublasHandle_t handle;
    cublasStatus_t cberr = cublasCreate_v2(&handle);
    if (cberr != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "Cannot create cublas handle: " << cberr << std::endl;
        return 1;
    }

    prof.tic("cublas");
    // Выполнить умножение матриц cdev := adev * bdev на GPU.
    double alpha = 1.0, beta = 0.0;
    cberr = cublasDgemm_v2(
            handle, CUBLAS_OP_T, CUBLAS_OP_T, n, n, n,
            &alpha, a_dev, n, b_dev, n, &beta, c_dev, n);
    prof.toc("cublas");

    if (cberr != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "Error launching cublasSgemm_v2: " << cberr << std::endl;
        return 1;
    }
    // Ожидать завершения операции.
    cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot synchronize kernel: " << cudaGetErrorString(cuerr) << std::endl;
        return 1;
    }
    // Удалить дексриптор CUBLAS.
    cberr = cublasDestroy_v2(handle);
    if (cberr != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "Cannot destroy cublas handle: " << cberr << std::endl;
        return 1;
    }

    prof.report();

    return 0;
}
