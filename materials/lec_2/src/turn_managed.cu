#include <iostream>
#include <vector>
#include <random>
#include <functional>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "profiler.h"
#include "turn.cuh"


int main()
{
    profiler prof;
    const size_t n = 1 << 27;
    float *x = nullptr, *y = nullptr, *tmp = nullptr, *angle = nullptr;

    cudaMallocManaged(&x, n * sizeof(float));
    cudaMallocManaged(&y, n * sizeof(float));
    cudaMallocManaged(&tmp, 4 * sizeof(float));
    cudaMallocManaged(&angle, sizeof(float));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0., 1.);

    for (auto i = 0; i < n; i++)
    {
        x[i] = dis(gen);        
        y[i] = dis(gen);
    }
    (*angle) = 45.;

    tmp[0] = cos((*angle));
    tmp[1] = -sin((*angle));
    tmp[2] = sin((*angle));
    tmp[3] = cos((*angle));

    prof.tic("cpu turn");
#pragma omp parallel for num_threads(8)    
    for(int i = 0; i < n; ++i)
    {
        float tmp_x = x[i] * cos(*angle) - y[i] * sin(*angle);
        float tmp_y = x[i] * sin(*angle) + y[i] * cos(*angle);

        x[i] = tmp_x;
        y[i] = tmp_y;
    }
    prof.toc("cpu turn");

    int threads = 1024;
    int blocks = n / threads;
    
    prof.tic("gpu turn global memory");
    for(int i = 0; i < 10; ++i)
    {
        turn_kernel<<<blocks, threads>>>(x, y, tmp);
        cudaCheckError( cudaGetLastError() );
        cudaCheckError( cudaDeviceSynchronize() );
    }
    prof.toc("gpu turn global memory");
    
    prof.tic("gpu turn constant memory");
    for(int i = 0; i < 10; ++i)
    {
        turn_kernel<<<blocks, threads>>>(x, y);
        cudaCheckError( cudaGetLastError() );
        cudaCheckError( cudaDeviceSynchronize() );
    }
    prof.toc("gpu turn constant memory");

    cudaFree(x);
    cudaFree(y);
    cudaFree(tmp);
    cudaFree(angle);

    prof.report();

    return 0;
}
