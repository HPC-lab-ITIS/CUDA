using namespace cooperative_groups;

template <int tile_sz>
__device__ float reduce_sum_tile_shfl(thread_block_tile<tile_sz> g, float val)
{
    int lane = g.thread_rank();

    for (int i = g.size() / 2; i > 0; i /= 2) 
        val += g.shfl_down(val, i);

    return val; 
}

template <int tile_sz>
__global__ void sum_kernel_tile_shfl(float *sum, float *input, int n)
{
    float my_sum = thread_sum(input, n);

    auto tile = tiled_partition<tile_sz>(this_thread_block());
    float tile_sum = reduce_sum_tile_shfl<tile_sz>(tile, my_sum);

    if (tile.thread_rank() == 0) 
        atomicAdd(sum, tile_sum);
}


float test_reduce_vector_cg_tile_shuffle(std::vector<float> &a, const size_t n, profiler &prof)
{
    count++;

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

    prof.tic("gpu reduction vector cg tiled shuffle");
    for(auto i = 0; i < test_runs; ++i)
    {
        cudaMemset(sum_dev, 0, sizeof(float));
        sum_kernel_tile_shfl<32><<<blocks, block_size>>>(sum_dev, a_dev, n);
        cudaDeviceSynchronize();
        cuerr = cudaGetLastError();
        cudaCheckError(cuerr);  
    }
    prof.toc("gpu reduction vector cg tiled shuffle");

    float tmp = 0;
    cudaMemcpy( &tmp, sum_dev, sizeof(float), cudaMemcpyDeviceToHost );

    cudaFree(a_dev);
    cudaFree(sum_dev);

    return tmp;
}
