__global__ void transpose_shared_kernel(float *a, const float *b)
{
    __shared__ float tile[block_size][block_size];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int m = blockDim.x * gridDim.x;

    tile[threadIdx.y][threadIdx.x] = b[x + y * m];

    __syncthreads();

    x = blockIdx.y * blockDim.x + threadIdx.x;
    y = blockIdx.x * blockDim.y + threadIdx.y;

    a[y * m + x] = tile[threadIdx.x][threadIdx.y];
}


int transpose_shared(std::vector<float> &a, std::vector<float> &b, const int n, profiler &prof)
{
    std::vector<float> tmp(n * n);
    auto n_bytes = n * n * sizeof(float);
    float *a_dev = nullptr, *b_dev = nullptr;

    cudaCheckError( cudaSetDevice(1) );

    cudaCheckError( cudaMalloc( &a_dev, n_bytes ) );

    cudaCheckError( cudaMalloc( &b_dev, n_bytes ) );

    dim3 threads(block_size, block_size, 1);
    dim3 blocks(n / block_size, n / block_size, 1);

    cudaCheckError( cudaMemcpy(b_dev, a.data(), n_bytes, cudaMemcpyHostToDevice) );

    prof.tic("shared memory transpose gpu");
    for(int i = 0; i < test_runs; ++i)
    {
        transpose_shared_kernel<<<blocks, threads>>>(a_dev, b_dev);
        cudaCheckError( cudaGetLastError() );
        cudaCheckError( cudaDeviceSynchronize() );
    }
    prof.toc("shared memory transpose gpu");

    cudaCheckError( cudaMemcpy(tmp.data(), a_dev, n_bytes, cudaMemcpyDeviceToHost) );
    
    std::cout << "shared memory transpose: ";
    std::cout << (std::equal(tmp.begin(), tmp.end(), b.begin()) ? "correct" : "wrong") << std::endl;

    cudaFree(a_dev);
    cudaFree(b_dev);

    return 0;
}
