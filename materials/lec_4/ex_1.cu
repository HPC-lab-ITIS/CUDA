#include <iostream>
#include <algorithm>
#include <mpi.h>

__global__ void kernel(int i, double *a)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    a[idx] = i;
}

int main(int argc, char *argv[])
{
    int rank; //process rank
    int size; //number of processes

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto count = 0;
    auto n = 1024;
    auto n_bytes = n * sizeof(double);
    
    cudaGetDeviceCount(&count);

    double *a;

    cudaSetDevice(rank);

    cudaMallocManaged( (void **)&a, n_bytes );

    kernel<<<n / 256, 256>>>(rank, a);

    cudaDeviceSynchronize();

    std::cout << std::accumulate(a, a + n, 0.0) << std::endl;

    cudaFree(a);

    MPI_Finalize();

    return 0;
}
