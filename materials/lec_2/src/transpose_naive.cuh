int cudaCheckError(cudaError_t cuerr)
{
    if (cuerr != cudaSuccess)
    {
        std::cout << cudaGetErrorString(cuerr);
        return 1;
    }

    return 0;
}


__global__ void transpose_naive_kernel(float *a, const float *b)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int m = blockDim.x * gridDim.x;

    a[x * m + y] = b[y * m + x];
}


int transpose_naive(std::vector<float> &a, std::vector<float> &b, const int n, profiler &prof)
{
    auto n_bytes = n * n * sizeof(float);
    float *a_dev = nullptr, *b_dev = nullptr;
    std::vector<float> tmp(n * n);

    cudaCheckError( cudaSetDevice(1) );

    cudaCheckError( cudaMalloc( &a_dev, n_bytes ) );

    cudaCheckError( cudaMalloc( &b_dev, n_bytes ) );

    dim3 threads(block_size, block_size, 1);
    dim3 blocks(n / block_size, n / block_size, 1);

    cudaCheckError( cudaMemcpy(b_dev, a.data(), n_bytes, cudaMemcpyHostToDevice) );

    prof.tic("naive transpose gpu");
    for(int i = 0; i < test_runs; ++i)
    {
        transpose_naive_kernel<<<blocks, threads>>>(a_dev, b_dev);
        cudaCheckError( cudaGetLastError() );
        cudaCheckError( cudaDeviceSynchronize() );
    }
    prof.toc("naive transpose gpu");

    cudaCheckError( cudaMemcpy(tmp.data(), a_dev, n_bytes, cudaMemcpyDeviceToHost) );

    std::cout << "naive transpose: ";
    std::cout << (std::equal(tmp.begin(), tmp.end(), b.begin()) ? "correct" : "wrong") << std::endl;

    cudaFree(a_dev);
    cudaFree(b_dev);

    return 0;
}
