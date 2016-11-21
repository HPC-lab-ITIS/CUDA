#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "profiler.h"

template <typename T>
    struct square
{
    __host__ __device__ T operator()(const T& x) const
        {
            return x * x;
        }
};

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

    square<double> unary_op;
    thrust::plus<double> binary_op;
    double init = 0.;

    prof.tic("thrust");
    double norm_gpu = std::sqrt( thrust::transform_reduce( X.begin(),X.end(), unary_op, init, binary_op ) );
    prof.toc("thrust");

    prof.tic("cpu");
    double norm_cpu = sqrt( std::inner_product(a.begin(), a.end(), a.begin(),  0.0) );
    prof.toc("cpu");

    if( fabs( norm_cpu - norm_gpu ) > 1e-5)
    {
        std::cout << "Wrong calculation" << std::endl;
        return 1;
    }

    prof.report();

    return 0;
}
