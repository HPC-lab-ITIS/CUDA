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
    std::vector<double> x(n);
    std::vector<double> y(n);


    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::mt19937 engine; // Mersenne twister MT19937
    auto generator = std::bind(distribution, engine);
    std::generate_n(x.begin(), n, generator);
    std::generate_n(y.begin(), n, generator);
    
    float angle = 45.;

    prof.tic("cpu turn");
#pragma omp parallel for num_threads(8)    
    for(int i = 0; i < n; ++i)
    {
        float tmp_x = x[i] * cos(angle) - y[i] * sin(angle);
        float tmp_y = x[i] * sin(angle) + y[i] * cos(angle);

        x[i] = tmp_x;
        y[i] = tmp_y;
    }
    prof.toc("cpu turn");

    turn(x, y, angle, prof);

    prof.report();

    return 0;
}
