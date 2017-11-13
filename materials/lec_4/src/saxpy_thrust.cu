#include <iostream>
#include <vector>
#include <random>
#include <functional>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include "profiler.h"


struct saxpy_functor
{
    const float a;
    
    saxpy_functor(float init) : a(init) {}

	__host__ __device__ float operator()(const float &x, const float &y) const
	{
		return a * x + y;
	}
};

void saxpy_fast(float a, thrust::device_vector<float> &x, thrust::device_vector<float> &y)
{
    thrust::transform(x.begin(), x.end(), y.begin(), y.begin(), saxpy_functor(a));
}


void saxpy_slow(float a, thrust::device_vector<float> &x, thrust::device_vector<float> &y)
{
    thrust::device_vector<float> tmp(x.size());

    thrust::fill(tmp.begin(), tmp.end(), a);

    thrust::transform(x.begin(), x.end(), tmp.begin(), tmp.begin(), thrust::multiplies<float>());

    thrust::transform(tmp.begin(), tmp.end(), y.begin(), y.begin(), thrust::plus<float>());

}

int check_result(std::vector<float> &x, std::vector<float> &y)
{
    for(auto i = 0; i < x.size(); ++i)
        if( fabs( x[i] - y[i] ) > 1e-5)
        {
            std::cout << "wrong result" << std::endl;
            return 1;
        }
            
    std::cout << "correct result" << std::endl;

    return 0;
}

int main(void)
{
    size_t n = 1 << 27;

    std::vector<float> x(n);
    std::vector<float> y(n);
    std::vector<float> check(n);
    
    profiler prof;

    std::uniform_real_distribution<> distribution(0.0, 1.0);
    std::mt19937 engine; 
    auto generator = std::bind(distribution, engine);

    std::generate(x.begin(), x.end(), generator);
    std::generate(y.begin(), y.end(), generator);
    float a = distribution(engine);

    thrust::device_vector<float> x_dev(x.begin(), x.end());
    thrust::device_vector<float> y_dev(y.begin(), y.end());


    prof.tic("slow saxpy");
    saxpy_slow(a, x_dev, y_dev);
    prof.toc("slow saxpy");

    thrust::copy(y_dev.begin(), y_dev.end(), check.begin());
    thrust::copy(y.begin(), y.end(), y_dev.begin());

    for(auto i = 0; i < n; ++i)
        y[i] += a * x[i];

    check_result(check, y);

    prof.tic("fast saxpy");
    saxpy_fast(a, x_dev, y_dev);
    prof.toc("fast saxpy");

    thrust::copy(y_dev.begin(), y_dev.end(), check.begin());

    check_result(check, y); 

    prof.report();

    return 0;
}
