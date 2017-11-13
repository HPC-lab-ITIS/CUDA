#include <iostream>
#include <vector>
#include <random>
#include <functional>
#include <algorithm>
#include "profiler.h"


int main(void)
{
    size_t n = 1 << 27;

    std::vector<float> x(n);
    std::vector<float> y(n);
    std::vector<float> z(n);
    
    profiler prof;

    std::uniform_real_distribution<> distribution(0.0, 1.0);
    std::mt19937 engine; 
    auto generator = std::bind(distribution, engine);

    std::generate(x.begin(), x.end(), generator);
    std::generate(y.begin(), y.end(), generator);
    float a = distribution(engine);

    prof.tic("saxpy cpu");
    for(auto i = 0; i < n; ++i)
        z[i] = a * x[i] + y[i];
    prof.toc("saxpy cpu");

    prof.tic("saxpy openacc");
#pragma acc parallel loop 
    for(auto i = 0; i < n; ++i)
        z[i] = a * x[i] + y[i];
    prof.toc("saxpy openacc");
    
    prof.report();

    return 0;
}
