using namespace cooperative_groups;

__device__ float reduce_sum(thread_group g, float *temp, float val)
{
	int lane = g.thread_rank();

	for (int i = g.size() / 2; i > 0; i /= 2)
	{
		temp[lane] = val;
		g.sync(); 
		if(lane < i)
			val += temp[lane + i];
		g.sync(); 
	}
	return val; 
}


__device__ float thread_sum(float *input, const size_t n)
{
	float sum = 0;

	for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n/4; i += blockDim.x * gridDim.x)
	{
		float4 in = ((float4*)input)[i];
		sum += in.x + in.y + in.z + in.w;
	}

	return sum;
}


__global__ void sum_kernel_block(float *sum, float *input, const size_t n)
{
	float my_sum = thread_sum(input, n);

	extern __shared__ float temp[];

	auto g = this_thread_block();
	float block_sum = reduce_sum(g, temp, my_sum);

	if(g.thread_rank() == 0)
		atomicAdd(sum, block_sum);
}


float test_reduce_vector_cg(std::vector<float> &a, const size_t n, profiler &prof)
{
    count++;
    size_t shared_bytes = block_size * sizeof(float);

    float *a_dev = nullptr;
    float *sum_dev = nullptr;

    cudaError_t cuerr = cudaMalloc( (void**)&a_dev, n * sizeof(float));
    cudaCheckError(cuerr);

    cuerr = cudaMalloc( (void**)&sum_dev, sizeof(float));
    cudaCheckError(cuerr);

    cuerr = cudaMemcpy(a_dev, a.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cuerr);
    cudaMemset(sum_dev, 0, sizeof(float));

    cudaDeviceSynchronize();

    int blocks = n / (16 * block_size);

    prof.tic("gpu reduction vector cg");
    for(auto i = 0; i < test_runs; ++i)
    {
        cudaMemset(sum_dev, 0, sizeof(float));
        sum_kernel_block<<<blocks, block_size, shared_bytes>>>(sum_dev, a_dev, n);
        cudaDeviceSynchronize();
        cuerr = cudaGetLastError();
        cudaCheckError(cuerr);  
    }
    prof.toc("gpu reduction vector cg");

    float tmp = 0;
    cudaMemcpy( &tmp, sum_dev, sizeof(float), cudaMemcpyDeviceToHost );

    cudaFree(a_dev);
    cudaFree(sum_dev);

    return tmp;
}
