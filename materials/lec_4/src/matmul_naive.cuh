int cudaCheckError(cudaError_t cuerr)
{
    if (cuerr != cudaSuccess)
    {
        std::cout << cudaGetErrorString(cuerr);
        return 1;
    }
    
    return 0;
}

__global__ void matmult_naive(float* a, float* b, float* c, size_t n)
{
    float sum = 0.;

    //Смещение для a [i][0]
    int i =  threadIdx.y + blockDim.y * blockIdx.y;

    //Смещение для b [0][j]
    int j = threadIdx.x + blockDim.x * blockIdx.x;

    //Перемножить строку и столбец
    for (auto k = 0; k < n; k++)
        sum += a[i * n + k] * b[k * n + j];

    //Сохранить результат в глобальной памяти
    c[i * n + j] = sum;
}



void matmult_naive(std::vector<float> &a, std::vector<float> &b, std::vector<float> &c, profiler &prof)
{
    auto n_bytes = n * n * sizeof(float);
    float *a_dev = nullptr, *b_dev = nullptr, *c_dev = nullptr;

    cudaCheckError( cudaMalloc(&a_dev, n_bytes) );
    cudaCheckError( cudaMalloc(&b_dev, n_bytes) );
    cudaCheckError( cudaMalloc(&c_dev, n_bytes) );

    cudaCheckError( cudaMemcpy (a_dev, a.data(), n_bytes, cudaMemcpyHostToDevice) );
    cudaCheckError( cudaMemcpy (b_dev, b.data(), n_bytes, cudaMemcpyHostToDevice) );

    dim3 block(block_size, block_size);
    dim3 grid(n / block_size, n / block_size);

    prof.tic("mult naive");
    matmult_naive<<<grid, block>>>(a_dev, b_dev, c_dev, n);
    cudaCheckError( cudaGetLastError() );
    cudaCheckError( cudaDeviceSynchronize() );
    prof.toc("mult naive");

    cudaCheckError( cudaMemcpy (c.data(), c_dev, n_bytes, cudaMemcpyDeviceToHost) );

    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);
}
