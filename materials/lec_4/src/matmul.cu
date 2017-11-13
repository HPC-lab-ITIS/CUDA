const int n = 1 << 11;
const int block_size = 32;

#include <iostream>
#include <vector>
#include "profiler.h"
#include <algorithm>
#include <random>
#include <cublas_v2.h>
#include <functional>
#include "matmul_naive.cuh"
#include "matmul_shared.cuh"
#include "matmul_cublas.cuh"


int main()
{
    std::vector<float> a(n*n,0.);
    std::vector<float> b(n*n,0.);
    std::vector<float> c(n*n,0.);
    std::vector<float> c_host(n*n,0.);
    profiler prof;
    std::uniform_real_distribution<> distribution(0.0, 1.0);
    std::mt19937 engine; 
    auto generator = std::bind(distribution, engine);
    std::generate_n(a.begin(), n, generator);
    std::generate_n(b.begin(), n, generator);

    cudaSetDevice(1);

    matmult_naive(a, b, c, prof);
    
    matmult_shared(a, b, c, prof);
    
    matmult_cublas(a, b, c, prof);
    
    prof.tic("seq_mult");
#pragma omp parallel for
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < n; ++j)
            for(int k = 0; k < n; ++k)
                c_host[i * n + j] += a[i * n + k] * b[k * n + j];
    prof.toc("seq_mult");

    prof.report();

    for(auto i = 0; i < c.size(); ++i)
        if(fabs(c_host[i] - c[i]) / c_host[i] > 1e-5)
        {
            std::cout << "fail" << std::endl;
            return 1;
        }

    return 0;
}
