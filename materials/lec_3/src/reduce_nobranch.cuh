__global__ void reduce_nobranch(float* in, int n, int m)
{
    __shared__ float buf[1024];
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    buf[threadIdx.x] = in[n + i];

    __syncthreads();

    for(int step = 1; step < blockDim.x; step *= 2)
    {
        int index = 2 * step * threadIdx.x;
        if( index + step < blockDim.x )
            buf[index] += buf[index + step];

        __syncthreads();
    }

    if(threadIdx.x == 0)
        in[m + blockIdx.x] = buf[0];
}

float test_reduce_nobranch(std::vector<float> &a, const int n, profiler &prof)
{
    count++;
    float *a_dev = nullptr, *b_dev = nullptr;
    cudaError_t cuerr = cudaMalloc( (void**)&a_dev, 2 * n * sizeof(float));
    cudaMalloc( (void**)&b_dev, 2 * n * sizeof(float));
    cudaCheckError(cuerr);
    cuerr = cudaMemcpy ( b_dev, a.data(), n * sizeof(float), cudaMemcpyHostToDevice );
    cudaCheckError(cuerr); 

    auto threads = 1024;
    auto blocks = n / threads;
    auto first = 0;
    auto second = n;
    auto leftover = 0;

    cudaDeviceSynchronize();

    prof.tic("gpu reduction without branching");
    for(auto ii = 0; ii < test_runs; ++ii)
    {
        first = 0;
        second = n;
        leftover = 0;
        cudaMemcpy( a_dev, b_dev, n * sizeof(float), cudaMemcpyDeviceToDevice );
        cudaMemset(a_dev + second, 0, n * sizeof(float));

        for(auto i = blocks; i > 1; i /= threads)
        {
            reduce_nobranch<<<i, threads>>>(a_dev, first, second);
            cuerr = cudaGetLastError();
            cudaCheckError(cuerr);        
            std::swap(first, second);
            leftover = i;
        }

        cudaDeviceSynchronize();    

        reduce_nobranch<<<1, leftover>>>(a_dev, first, second);

        cudaDeviceSynchronize();
        cuerr = cudaGetLastError();
        cudaCheckError(cuerr);    
    }
    prof.toc("gpu reduction without branching");
    
    float sum_dev = 0;
    cuerr = cudaMemcpy( &sum_dev, a_dev + second, sizeof(float), cudaMemcpyDeviceToHost );
    cudaCheckError(cuerr);  

    cudaFree(a_dev);
    
    return sum_dev;
}
