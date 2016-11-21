#include <iostream>
#include <algorithm>
#include <omp.h>

__global__ void kernel(int i, double *a)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    a[idx] = i;
}

int main()
{
    auto count = 0;
    auto n = 1024;
    auto n_bytes = n * sizeof(double);
    cudaGetDeviceCount(&count);

    double *a[count];

#pragma omp parallel num_threads(count)
    {
        auto i = omp_get_thread_num(); 
        cudaSetDevice(i);
        cudaMallocManaged( (void **)&a[i], n_bytes );
        kernel<<<n / 256, 256>>>(i, a[i]);
        cudaDeviceSynchronize();
        std::cout << "thread #" << i << " has sum = " << std::accumulate(a[i], a[i] + n, 0.0) << std::endl;
        cudaFree(a[i]);
    }
    
    return 0;
}
