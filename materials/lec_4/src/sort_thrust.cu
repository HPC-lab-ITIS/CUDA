#include <iostream>
#include <random>
#include <vector>
#include "profiler.h"
#include <algorithm>
#include <functional>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

int main(void)
{
    cudaSetDevice(1);

    const size_t n = 1 << 27;
    profiler prof;
    std::vector<float> a(n);

    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    std::mt19937 engine;
    auto generator = std::bind(distribution, engine);
    std::generate_n(a.begin(), n, generator);
    
    prof.tic("sort cpu");
    std::sort(a.begin(), a.end());
    prof.toc("sort cpu");

    prof.tic("thrust data transfer");
    thrust::device_vector<float> a_dev(a.begin(), a.end());
    prof.toc("thrust data transfer");
   
    prof.tic("sort thrust");
    thrust::reduce(a_dev.begin(), a_dev.end());
    prof.toc("sort thrust");

    prof.report();

    return 0;
}
