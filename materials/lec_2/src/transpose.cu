const int block_size = 32;
const int test_runs = 32;


#include <iostream>
#include <vector>
#include <random>
#include <functional>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "profiler.h"
#include "transpose_naive.cuh"
#include "transpose_shared.cuh"


int main(int argc, char *argv[])
{
    profiler prof;
    const int n = 1 << 14;
    std::vector<float> a(n * n);
    std::vector<float> b(n * n);

    std::uniform_real_distribution<> distribution(0.0, 1.0);
    std::mt19937 engine; 
    auto generator = std::bind(distribution, engine);
    std::generate_n(a.begin(), n * n, generator);

    prof.tic("cpu transpose");
#pragma omp parallel for num_threads(8)
    for(auto i = 0; i < n; ++i)
        for(auto j = 0; j < n; ++j)
            b[j + i * n] = a[i + j * n];
    prof.toc("cpu transpose");

    transpose_naive(a, b, n, prof);

    transpose_shared(a, b, n, prof);

    prof.report();

    return 0;
}
