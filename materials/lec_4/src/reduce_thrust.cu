#include <iostream>
#include <random>
#include <vector>
#include "profiler.h"
#include <algorithm>
#include <functional>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

const auto test_runs = 1024 / 4;

int main(void)
{
    cudaSetDevice(1);

    const size_t n = 1 << 25;
    profiler prof;
    std::vector<float> a(n);

    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    std::mt19937 engine;
    auto generator = std::bind(distribution, engine);
    std::generate_n(a.begin(), n, generator);
    
    float sum = 0, sum_dev = 0;

    prof.tic("reduction cpu");
    for(auto i = 0; i < test_runs; ++i)
        sum = std::accumulate(a.begin(), a.end(), 0.0);
    prof.toc("reduction cpu");

    thrust::device_vector<float> a_dev(a.begin(), a.end());
   
    prof.tic("reduction thrust");
    for(auto i = 0; i < test_runs; ++i)
        sum_dev = thrust::reduce(a_dev.begin(), a_dev.end(), 0., thrust::plus<float>());
    prof.toc("reduction thrust");

    std::cout << "Error: " << fabs(sum - sum_dev) / sum << std::endl;

    prof.report();

    return 0;
}
