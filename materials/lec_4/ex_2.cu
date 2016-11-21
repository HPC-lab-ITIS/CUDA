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

    int next = ( rank < (size - 1) ? (rank + 1) : 0);
    int prev = (rank > 0 ? (rank - 1) : size - 1);

    auto count = 0;
    auto n = 1024;
    auto n_bytes = n * sizeof(double);

    cudaGetDeviceCount(&count);

    double *a, *b;

    cudaSetDevice(rank);

    cudaMallocManaged( (void **)&a, n_bytes );
    cudaMallocManaged( (void **)&b, n_bytes );

    kernel<<<n / 256, 256>>>(rank, a);

    cudaDeviceSynchronize();

    MPI_Request request; 
    MPI_Isend(a, n, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &request);
    MPI_Recv(b, n, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    std::cout << "rank = " << rank << "; sum = " << std::accumulate(b, b + n, 0.0) << std::endl;

    cudaFree(a);
    cudaFree(b);

    MPI_Finalize();

    return 0;
}
