#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "profiler.h"

struct func
{
	__host__ __device__ double operator()(double x, double y)
	{
		return sin(x) + cos(y)*cos(y);
	}
};

int main()
{
    int n = 1024 * 1024 * 128;

    profiler prof;

    std::vector<double> a(n);
    std::vector<double> b(n);
    std::vector<double> c_cpu(n);
    std::vector<double> c_gpu(n);

    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::mt19937 engine; // Mersenne twister MT19937
    auto generator = std::bind(distribution, engine);

    std::generate_n(a.begin(), n, generator);
    std::generate_n(b.begin(), n, generator);

    thrust::device_vector<double> X(a.begin(), a.end());
    thrust::device_vector<double> Y(b.begin(), b.end());
    thrust::device_vector<double> Z(n);

    prof.tic("thrust");
    for(auto i = 0; i < 100; ++i)
        thrust::transform( 
                thrust::make_zip_iterator( thrust::make_tuple( X.begin(), Y.begin() ) ), 
                thrust::make_zip_iterator( thrust::make_tuple( X.end(), Y.end() ) ),
                Z.begin(),
                []__device__(thrust::tuple<double, double> t)
                { double x,y; thrust::tie(x,y) = t; return sin(x) + cos(y) * cos(y);} );
    prof.toc("thrust");

    thrust::copy(Z.begin(), Z.end(), c_gpu.begin());

    prof.tic("cpu sum");
    for(auto i = 0; i < n; ++i)
        c_cpu[i] = sin(a[i]) + cos(b[i])*cos(b[i]);
    prof.toc("cpu sum");


    for(auto i = 0; i < n; ++i)
        if( fabs( c_cpu[i] - c_gpu[i] ) > 1e-5)
        {
            std::cout << "Wrong calculation" << std::endl;
            return 1;
        }

    prof.report();

    return 0;
}
