__inline__ __device__ float warpReduceSum(float val)
{
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down(val, offset);

    return val;
}

__inline__ __device__ float blockReduceSum(float val)
{
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val); 

    if (lane==0)
        shared[wid]=val;

    __syncthreads();             

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid==0)
        val = warpReduceSum(val);

    return val;
}

__global__ void deviceReduceKernel(float *in, float* out, const int N)
{
    float sum = 0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        sum += in[i];

    sum = blockReduceSum(sum);
    if (threadIdx.x==0)
        out[blockIdx.x] = sum;
}

float test_reduce_shuffle(std::vector<float> &a, const int n, profiler &prof)
{
    count++;
    float *a_dev = nullptr;
    float *tmp = nullptr;
    float sum_dev = 0;
    int grid_size = n / block_size;

    cudaError_t cuerr = cudaMalloc((void**)&tmp, grid_size * sizeof(float));
    cudaCheckError(cuerr);
    
    cuerr = cudaMalloc((void**)&a_dev, n * sizeof(float));
    cudaCheckError(cuerr);
    
    cuerr = cudaMemcpy(a_dev, a.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cuerr);
    
    cudaDeviceSynchronize();

    prof.tic("gpu reduction simple shuffle");
    for(auto i = 0; i < test_runs; ++i)
    {
	deviceReduceKernel<<<grid_size, block_size>>>(a_dev, tmp, n);    
	cudaDeviceSynchronize();
	cuerr = cudaGetLastError();
	cudaCheckError(cuerr); 

	deviceReduceKernel<<<1, 1024>>>(tmp, tmp, grid_size);
        cudaDeviceSynchronize();
	cuerr = cudaGetLastError();
	cudaCheckError(cuerr); 
    }
    prof.toc("gpu reduction simple shuffle");

    cuerr = cudaMemcpy(&sum_dev, tmp, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError(cuerr);  
    
    cudaFree(tmp);
    cudaFree(a_dev);

    return sum_dev;
}
