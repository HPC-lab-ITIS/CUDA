#include <iostream>
#include <cmath>
#include <algorithm> 

__global__ void sum_kernel(size_t n, double *a, double *b, double *c)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t grid_size = blockDim.x * gridDim.x;
    for(size_t i = idx; i < n; i += grid_size)
        c[i] = sin(a[i]) + cos(b[i])*cos(b[i]);
}
 
int main() 
{
    const size_t n = 1024*1024*128;
    double* a; 
    double* b; 
    double* c; 
    cudaMallocManaged(&a, n * sizeof(double));
    cudaMallocManaged(&b, n * sizeof(double));
    cudaMallocManaged(&c, n * sizeof(double));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0., 1.);

    for (auto i = 0; i < n; i++)
    {
        a[i] = dis(gen);        
        b[i] = dis(gen);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for(auto i = 0; i < 1; ++i)
        sum_kernel<<<208, 256>>>(n, a, b, c);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop);

    std::cout << "Elapsed time: " << gpu_time << " ms." << std::endl;

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
