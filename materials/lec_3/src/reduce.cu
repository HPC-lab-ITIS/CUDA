const auto block_size = 1024;
const auto test_runs = 1024 / 4;
auto count = 0;

#include <iostream>
#include <random>
#include <vector>
#include "profiler.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cooperative_groups.h>
#include <functional>
#include "reduce_naive.cuh"
#include "reduce_shared.cuh"
#include "reduce_nobranch.cuh"
#include "reduce_noconflict.cuh"
#include "reduce_largeblock.cuh"
#include "reduce_shuffle.cuh"
#include "reduce_shuffle_warp.cuh"
#include "reduce_shuffle_block.cuh"
#include "reduce_vector_cg.cuh"
#include "reduce_vector_cg_tile.cuh"


int main(int argc, char **argv)
{
    int device = 0;

    if (argc>1)
    {
        std::istringstream iss(argv[1]);
        iss >> device;
    }

    cudaSetDevice(device);

    const size_t n = 32 * 1024 * 1024;
    profiler prof;
    std::vector<float> a(n);

    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    std::mt19937 engine;
    auto generator = std::bind(distribution, engine);
    std::generate_n(a.begin(), n, generator);

    float sum = 0, sum_dev = 0;

    prof.tic("reduction cpu");
    for(auto i = 0; i < test_runs; ++i)
        sum += std::accumulate(a.begin(), a.end(), 0.0);
    prof.toc("reduction cpu");

    sum /= test_runs;

    sum_dev += test_reduce_naive(a, n, prof);

    sum_dev += test_reduce_shared(a, n, prof);

    sum_dev += test_reduce_nobranch(a, n, prof);

    sum_dev += test_reduce_largeblock(a, n, prof);

    sum_dev += test_reduce_shuffle(a, n, prof);

    sum_dev += test_reduce_shuffle_warp(a, n, prof);

    sum_dev += test_reduce_shuffle_block(a, n, prof);

    sum_dev += test_reduce_vector_cg(a, n, prof);

    sum_dev += test_reduce_vector_cg_tile(a, n, prof);

    std::cout << "Error: " << fabs(sum - sum_dev/count) / sum << std::endl;

    prof.report();

    return 0;
}
