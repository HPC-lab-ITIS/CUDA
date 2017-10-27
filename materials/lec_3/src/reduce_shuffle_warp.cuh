__global__ void deviceReduceWarpAtomicKernel(float *in, float *out, int N)
{
    float sum = 0;

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        sum += in[i];
    }

    sum = warpReduceSum(sum);

    if( (threadIdx.x % warpSize) == 0)
        atomicAdd(out, sum);
}


float test_reduce_shuffle_warp(std::vector<float> &a, const int n, profiler &prof)
{
    count++;
    float *a_dev = nullptr;
    float *tmp_at = nullptr;
    float sum_dev = 0;
    int threads = block_size;
    int blocks = 128;
    
    cudaError_t cuerr = cudaMalloc((void**)&tmp_at, sizeof(float));
    cudaCheckError(cuerr);
    
    cuerr = cudaMalloc((void**)&a_dev, n * sizeof(float));
    cudaCheckError(cuerr);
    
    cuerr = cudaMemcpy(a_dev, a.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cuerr);
    
    cudaMemset(tmp_at, 0, sizeof(float));

    cudaDeviceSynchronize();

    prof.tic("gpu reduction warp shuffle");
    for(auto i = 0; i < test_runs; ++i)
    {
        cudaMemset(tmp_at, 0, sizeof(float));
	deviceReduceWarpAtomicKernel<<<blocks, threads>>>(a_dev, tmp_at, n);
	cudaDeviceSynchronize();
	cuerr = cudaGetLastError();
	cudaCheckError(cuerr);  
    }
    prof.toc("gpu reduction warp shuffle");

    cudaMemcpy(&sum_dev, tmp_at, sizeof(float), cudaMemcpyDeviceToHost);    
    
    cudaFree(tmp_at);
    cudaFree(a_dev);

    return sum_dev;
}
