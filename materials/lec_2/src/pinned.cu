#include <iostream>
#include <random>
#include <functional>
#include <vector>
#include <algorithm>
#include "profiler.h"
#include "pinned_copy.cuh"
#include "zero_copy.cuh"

int main()
{
    int n = 1 << 18;
    std::vector<float> a(n);
    profiler prof_1, prof_2;

    std::uniform_real_distribution<> distribution(0.0, 1.0);
    std::mt19937 engine; 
    auto generator = std::bind(distribution, engine);
    std::generate_n(a.begin(), n, generator);

    zero_copy(a, prof_1);
    prof_1.report();

    pinned_copy(a, prof_2);
    prof_2.report();

    return 0;
}
