__global__ void reduce_largeblock(float *in_data, float *out_data)
{
    __shared__ float buf[block_size];

    int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    buf[threadIdx.x] = in_data[i] + in_data[i + blockDim.x];

    __syncthreads ();

    for (int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if ( threadIdx.x < s )
            buf[threadIdx.x] += buf[threadIdx.x + s];

        __syncthreads();
    }

    if( threadIdx.x == 0 )
        out_data[blockIdx.x] = buf[threadIdx.x];
}

float test_reduce_largeblock(std::vector<float> &a, const int n, profiler &prof)
{
    count++;
    float *a_dev[2];
    float *b_dev = nullptr;

    cudaError_t cuerr = cudaMalloc( (void**)&b_dev, 2 * n * sizeof(float));
    cudaCheckError(cuerr);

        
    cuerr = cudaMemcpy ( b_dev, a.data(), n * sizeof(float), cudaMemcpyHostToDevice );
    cudaCheckError(cuerr); 

    cudaMalloc( (void**)&a_dev[0], n * sizeof(float));
    cudaCheckError(cuerr);

    cuerr = cudaMalloc( (void**)&a_dev[1], n * sizeof(float));
    cudaCheckError(cuerr);



    auto i = 0;
    auto j = 1;
    float sum_dev = 0;

    prof.tic("gpu reduction double blocks");
    for(auto ii = 0; ii < test_runs; ++ii)
    {   
        i = 0;
        j = 1;
        cudaMemcpy( a_dev[0], b_dev, n * sizeof(float), cudaMemcpyHostToDevice );
        cudaMemset(a_dev[1], 0, n * sizeof(float));

        for (j = n; j >= block_size; j /= (2 * block_size), i ^= 1)
        {
            auto threads = block_size;
            auto blocks = j / (2 * block_size);

            reduce_largeblock <<<blocks, threads>>>(a_dev[i], a_dev[i ^ 1]);
            cuerr = cudaGetLastError();
            cudaCheckError(cuerr);
        }

        std::vector<float> b(j);
        cuerr = cudaMemcpy(b.data(), a_dev[i], sizeof(float) * j, cudaMemcpyDeviceToHost);
        cudaCheckError(cuerr);

        sum_dev = std::accumulate(b.begin(), b.end(), 0.0);
    }
    prof.toc("gpu reduction double blocks");

    cudaFree(a_dev[0]);
    cudaFree(a_dev[1]);

    return sum_dev;
}
