#include <iostream>
#include "profiler.h"
#include <vector>

__global__ void kernel(double *a)
{
	size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	a[i] += i;
}


int main()
{
    size_t n = 128*1024;
    size_t n_bytes = n*sizeof(double);
    double *a_dev = nullptr, *buffer = nullptr, *buff_map = nullptr;
    std::vector<double> a_host(n,0.);
    profiler prof;

    cudaSetDeviceFlags(cudaDeviceMapHost); 
    cudaHostAlloc(&buffer, n_bytes, cudaHostAllocMapped);

    for(auto i = 0; i < n; ++i)
    {
        a_host[i] = i;
        buffer[i] = i;
    }

    cudaHostGetDevicePointer(&buff_map, buffer, 0);
    
    cudaError_t cuerr = cudaMalloc( (void**)&a_dev, n_bytes);
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot allocate GPU memory" << cudaGetErrorString(cuerr);
        return 1;
    }    

    prof.tic("Zero copy");
    cudaThreadSynchronize();
    for(auto i = 0; i < 1000; ++i)
    {    
        kernel<<<13, 256>>>(buff_map);
        cuerr = cudaGetLastError();
        if (cuerr != cudaSuccess)
        {
            std::cout << "Cannot launch CUDA kernel " << cudaGetErrorString(cuerr);
            return 1;
        }

        cuerr = cudaDeviceSynchronize();
        if (cuerr != cudaSuccess)
        {
            std::cout << "Cannot synchronize CUDA kernel " << cudaGetErrorString(cuerr);
            return 1;
        }

    }
    cudaThreadSynchronize();
    prof.toc("Zero copy");

    prof.tic("Standart copy");
    cudaThreadSynchronize();
    for(auto i = 0; i < 1000; ++i)
    {    
        cuerr = cudaMemcpy( a_dev, a_host.data(), n_bytes, cudaMemcpyHostToDevice );

        if (cuerr != cudaSuccess)
        {
            std::cout << "Cannot copy data to device" << cudaGetErrorString(cuerr);
            return 1;
        }

        kernel<<<13, 256>>>(a_dev);
        cuerr = cudaGetLastError();
        if (cuerr != cudaSuccess)
        {
            std::cout << "Cannot launch CUDA kernel " << cudaGetErrorString(cuerr);
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
    prof.toc("Standart copy");

    for(auto i = 0; i < n; ++i)
        if(fabs(a_host[i] - buffer[i])>1e-5)
        {
            std::cout << "fail" << std::endl;
            return 1;
        }

    cudaFree(a_dev);
    cudaFreeHost(buffer);

    prof.report();

    return 0;
}
