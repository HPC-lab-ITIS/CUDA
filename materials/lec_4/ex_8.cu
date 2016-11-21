#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "profiler.h"

int main()
{
    int n = 1024 * 1024 * 128;

    profiler prof;

    std::vector<double> a(n);

    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::mt19937 engine; // Mersenne twister MT19937
    auto generator = std::bind(distribution, engine);

    std::generate_n(a.begin(), n, generator);

    thrust::device_vector<double> X(a.begin(), a.end());

    prof.tic("thrust");
    double sum_gpu = thrust::reduce(X.begin(),X.end());
    prof.toc("thrust");

    prof.tic("cpu sum");
    double sum_cpu = std::accumulate(a.begin(), a.end(), 0.0);
    prof.toc("cpu sum");


    if( fabs( sum_cpu - sum_gpu ) > 1e-5)
    {
        std::cout << "Wrong calculation" << std::endl;
        return 1;
    }

    prof.report();

    return 0;
}
