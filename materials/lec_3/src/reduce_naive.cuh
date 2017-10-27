__global__ void reduce_naive(float *in, int n, int m)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    for(int step = 1; step < blockDim.x; step *= 2)
    {
        if( ( threadIdx.x % (2*step) == 0) && (step + threadIdx.x < blockDim.x ) )
            in[n + i] += in[n + i + step];

        __syncthreads();
    }

    if(threadIdx.x == 0)
        in[m + blockIdx.x] = in[n + blockIdx.x*blockDim.x];
}

int cudaCheckError(cudaError_t cuerr)
{
    if (cuerr != cudaSuccess)
    {
        std::cout << cudaGetErrorString(cuerr);
        return 1;
    }
    
    return 0;
}

float test_reduce_naive(std::vector<float> &a, const int n, profiler &prof)
{
    count++;
    float *a_dev = nullptr, *b_dev = nullptr;
    cudaError_t cuerr = cudaMalloc( (void**)&a_dev, 2 * n * sizeof(float));
    cuerr = cudaMalloc( (void**)&b_dev, n * sizeof(float));
    cudaCheckError(cuerr);
    cuerr = cudaMemcpy ( b_dev, a.data(), n * sizeof(float), cudaMemcpyHostToDevice );
    cudaCheckError(cuerr); 

    auto threads = 1024;
    auto blocks = n / threads;
    auto first = 0;
    auto second = n;
    auto leftover = 0;

    cudaDeviceSynchronize();

    prof.tic("gpu naive reduction");
    for(auto ii = 0; ii < test_runs; ++ii)
    {
        first = 0;
        second = n;
        leftover = 0;
        cudaMemcpy( a_dev, b_dev, n * sizeof(float), cudaMemcpyDeviceToDevice );
        cudaMemset(a_dev + second, 0, n * sizeof(float));

        for(auto i = blocks; i > 1; i /= threads)
        {
            reduce_naive<<<i, threads>>>(a_dev, first, second);
            cuerr = cudaGetLastError();
            cudaCheckError(cuerr);
            std::swap(first, second);
            leftover = i;
        }

        cudaDeviceSynchronize();

        reduce_naive<<<1, leftover>>>(a_dev, first, second);

        cudaDeviceSynchronize();

        cuerr = cudaGetLastError();
        cudaCheckError(cuerr);
    }
    prof.toc("gpu naive reduction");

    float sum_dev = 0;
    cuerr = cudaMemcpy( &sum_dev, a_dev + second, sizeof(float), cudaMemcpyDeviceToHost );
    cudaCheckError(cuerr);  

    cudaFree(a_dev);

    return sum_dev;
}
