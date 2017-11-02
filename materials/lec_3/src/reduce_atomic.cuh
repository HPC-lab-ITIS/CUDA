__global__ void reduce_atomic(float *in, float *out, const int n)
{
    __shared__ float buf[block_size];

    int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    if(i + blockDim.x < n)
        buf[threadIdx.x] = in[i] + in[i + blockDim.x];

    __syncthreads ();

    for (int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if ( threadIdx.x < s )
            buf[threadIdx.x] += buf[threadIdx.x + s];

        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(out, buf[0]);
}

float test_reduce_atomic(std::vector<float> &a, const int n, profiler &prof)
{
    count++;
    float *a_dev = nullptr, *tmp = nullptr;

    cudaError_t cuerr = cudaMalloc( (void**)&tmp, sizeof(float));
    cudaCheckError(cuerr);

    cuerr = cudaMalloc( (void**)&a_dev, n * sizeof(float));
    cudaCheckError(cuerr);

    cuerr = cudaMemcpy ( a_dev, a.data(), n * sizeof(float), cudaMemcpyHostToDevice );
    cudaCheckError(cuerr);

    cudaDeviceSynchronize();

    auto threads = block_size;
    auto blocks = n / (2 * block_size);

    prof.tic("gpu reduction atomic");
    for(auto ii = 0; ii < test_runs; ++ii)
    {   
        cudaMemset(tmp, 0, sizeof(float));
        reduce_atomic<<<blocks, threads>>>(a_dev, tmp, n);
        cuerr = cudaGetLastError();
        cudaCheckError(cuerr);
    }
    cudaDeviceSynchronize();
    prof.toc("gpu reduction atomic");

    float sum_dev;
    cudaMemcpy(&sum_dev, tmp, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(a_dev);
    cudaFree(tmp);

    return sum_dev;
}
