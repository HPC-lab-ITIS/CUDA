#include <iostream>
#include <vector>
#include "profiler.h"
#include <fstream>
#include <sstream>
#include <algorithm>

// Ядро, выполняется параллельно на большом числе нитей.
__global__ void sum_kernel(size_t n, double *a, double *b, double *c)
{
	// Глобальный индекс нити.
	size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	size_t grid_size = blockDim.x * gridDim.x;
	// Выполнить обработку соответствующих данной нити данных.
        for(size_t i = idx; i < n; i += grid_size)
            c[i] = sin(a[i]) + cos(b[i])*cos(b[i]);
}

int gpu_sum(std::vector<double> &a, std::vector<double> &b, std::vector<double> &c)
{
    auto n = a.size();
    auto nb = n * sizeof(double);
    double *aDev = nullptr, *bDev = nullptr, *cDev = nullptr;
    profiler prof_1;

    //Выделить память на GPU
    cudaError_t cuerr = cudaMalloc ( (void**)&aDev, nb );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot allocate GPU memory for aDev" << cudaGetErrorString(cuerr);
        return 1;
    }

    cuerr = cudaMalloc ( (void**)&bDev, nb );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot allocate GPU memory for bDev" << cudaGetErrorString(cuerr);
        return 1;
    }

    cuerr = cudaMalloc ( (void**)&cDev, nb );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot allocate GPU memory for cDev" << cudaGetErrorString(cuerr);
        return 1;
    }

    // Задать конфигурацию запуска n нитей
    size_t block_size = 1024;
    //size_t grid_size = n/block_size;

    // Скопировать входные данные из памяти CPU в память GPU.
    cuerr = cudaMemcpy ( aDev, a.data(), nb, cudaMemcpyHostToDevice );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot copy data from a to aDev" << cudaGetErrorString(cuerr);
        return 1;
    }

    cuerr = cudaMemcpy ( bDev, b.data(), nb, cudaMemcpyHostToDevice );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot copy data from a to bDev" << cudaGetErrorString(cuerr);
        return 1;
    }


    cudaThreadSynchronize();
    prof_1.tic("gpu_sum");
    for(auto i = 0; i < 100; ++i)
        // Вызвать ядро с заданной конфигурацией для обработки данных.
        sum_kernel<<<208, block_size>>>(n, aDev, bDev, cDev);
    cudaThreadSynchronize();
    prof_1.toc("gpu_sum");

    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot launch CUDA kernel " << cudaGetErrorString(cuerr);
        return 1;
    }

    // Ожидать завершения работы ядра.
    cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot synchronize CUDA kernel " << cudaGetErrorString(cuerr);
        return 1;
    }

    // Скопировать результаты в память CPU.
    cuerr = cudaMemcpy ( c.data(), cDev, nb, cudaMemcpyDeviceToHost );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot copy data from cdev to c " << cudaGetErrorString(cuerr);
        return 1;
    }

    // Освободить выделенную память GPU.
    cudaFree(aDev);
    cudaFree(bDev);
    cudaFree(cDev);
    
    prof_1.report();

    return 0;
}

int main(int argc, char *argv[])
{
    size_t n_par = 128;

    if (argc>1)
    {
        std::istringstream iss(argv[1]);
        iss >> n_par;
    }

    size_t n = n_par*1024*1024;

    profiler prof;
    std::vector<double> a(n);
    std::vector<double> b(n);
    std::vector<double> c_cpu(n);
    std::vector<double> c_gpu(n);

    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::mt19937 engine; // Mersenne twister MT19937
    auto generator = std::bind(distribution, engine);
    std::generate_n(a.begin(), n, generator);
    std::generate_n(b.begin(), n, generator);

    prof.tic("cpu sum");
    for(auto i = 0; i < n; ++i)
        c_cpu[i] = sin(a[i]) + cos(b[i])*cos(b[i]);
    prof.toc("cpu sum");

    gpu_sum(a, b, c_gpu);

    for(auto i = 0; i < n; ++i)
        if( fabs( c_cpu[i] - c_gpu[i] ) > 1e-5)
        {
            std::cout << "Wrong calculation" << std::endl;
            return 1;
        }

    prof.report();

    return 0;
}
