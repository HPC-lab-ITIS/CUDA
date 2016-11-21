#include <iostream>
#include <algorithm>

__global__ void kernel(int i, double *a)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    a[idx] = i;
}

int main()
{
    auto count = 13;
    auto n = 1024;
    auto n_bytes = n * sizeof(double);

    double *a[count];
    cudaStream_t streams[count];

    for (auto i = 0; i < count; ++i)
    {
        cudaStreamCreate(&streams[i]);
        cudaMallocManaged( (void **)&a[i], n_bytes );
        kernel<<<1, 1024, 0, streams[i]>>>(i, a[i]);
        cudaDeviceSynchronize();
        std::cout << std::accumulate(a[i], a[i] + n, 0.0) << std::endl;
        cudaStreamDestroy(streams[i]);
        cudaFree(a[i]);
    }

    return 0;
}
