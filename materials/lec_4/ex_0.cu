#include <iostream>
#include <algorithm>

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

    for (auto i = 0; i < count; ++i)
    {
        cudaSetDevice(i);
        cudaMallocManaged( (void **)&a[i], n_bytes );
        kernel<<<n / 256, 256>>>(i, a[i]);
        cudaDeviceSynchronize();
        std::cout << std::accumulate(a[i], a[i] + n, 0.0) << std::endl;
        cudaFree(a[i]);
    }

    return 0;
}
