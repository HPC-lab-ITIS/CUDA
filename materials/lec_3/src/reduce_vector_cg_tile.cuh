using namespace cooperative_groups;

template <typename group_t> __device__ float reduce_sum(group_t g, float *temp, float val)
{
	int lane = g.thread_rank();

#pragma unroll
	for (int i = g.size() / 2; i > 0; i /= 2)
	{
		temp[lane] = val;
		g.sync(); 
		if (lane < i)
			val += temp[lane + i];
		g.sync(); 

	return val;
}


__global__ void sum_kernel_tiled(float *sum, float *input, float n)
{
    float my_sum = thread_sum(input, n);

    extern __shared__ float temp[];

    auto g = this_thread_block();
    auto tileIdx = g.thread_rank() / 32;
    float *t = &temp[32 * tileIdx];

    thread_block_tile<32> tile32 = tiled_partition<32>(g);
    float tile_sum = reduce_sum(tile32, t, my_sum);

    if (tile32.thread_rank() == 0) 
        atomicAdd(sum, tile_sum);
}


float test_reduce_vector_cg_tile(std::vector<float> &a, const size_t n, profiler &prof)
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

    prof.tic("gpu reduction vector cg tiled");
    for(auto i = 0; i < test_runs; ++i)
    {
        cudaMemset(sum_dev, 0, sizeof(float));
        sum_kernel_tiled<<<blocks, block_size, shared_bytes>>>(sum_dev, a_dev, n);
        cudaDeviceSynchronize();
        cuerr = cudaGetLastError();
        cudaCheckError(cuerr);  
    }
    prof.toc("gpu reduction vector cg tiled");

    float tmp = 0;
    cudaMemcpy( &tmp, sum_dev, sizeof(float), cudaMemcpyDeviceToHost );

    cudaFree(a_dev);
    cudaFree(sum_dev);

    return tmp;
}
