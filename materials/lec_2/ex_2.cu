#include <iostream>
#include "profiler.h"
#include <vector>

int main()
{
    size_t n = 128*1024;
    size_t n_bytes = n*sizeof(double);
    double *a_dev = nullptr, *buffer = nullptr;
    std::vector<double> a_host(n,0.);
    profiler prof;

    cudaMallocHost( (void **) &buffer, n_bytes);

    for(auto i = 0; i < n; ++i)
    {
        a_host[i] = i;
        buffer[i] = i;
    }

    cudaError_t cuerr = cudaMalloc( (void**)&a_dev, n_bytes);
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot allocate GPU memory" << cudaGetErrorString(cuerr);
        return 1;
    }    

    prof.tic("Pinned memory");
    cudaThreadSynchronize();
    for(auto i = 0; i < 1000; ++i)
    {    
        cuerr = cudaMemcpy( a_dev, buffer, n_bytes, cudaMemcpyHostToDevice );

        if (cuerr != cudaSuccess)
        {
            std::cout << "Cannot copy data to device" << cudaGetErrorString(cuerr);
            return 1;
        }

        cuerr = cudaMemcpy ( buffer, a_dev, n_bytes, cudaMemcpyDeviceToHost );
        if (cuerr != cudaSuccess)
        {
            std::cout << "Cannot copy data from device" << cudaGetErrorString(cuerr);
            return 1;
        }
    }
    cudaThreadSynchronize();
    prof.toc("Pinned memory");

    prof.tic("Paged memory");
    cudaThreadSynchronize();
    for(auto i = 0; i < 1000; ++i)
    {    
        cuerr = cudaMemcpy( a_dev, a_host.data(), n_bytes, cudaMemcpyHostToDevice );

        if (cuerr != cudaSuccess)
        {
            std::cout << "Cannot copy data to device" << cudaGetErrorString(cuerr);
            return 1;
        }

        cuerr = cudaMemcpy ( a_host.data(), a_dev, n_bytes, cudaMemcpyDeviceToHost );
        if (cuerr != cudaSuccess)
        {
            std::cout << "Cannot copy data from device" << cudaGetErrorString(cuerr);
            return 1;
        }
    }
    cudaThreadSynchronize();
    prof.toc("Paged memory");

    cudaFree(a_dev);
    cudaFreeHost(buffer);

    prof.report();

    return 0;
}
