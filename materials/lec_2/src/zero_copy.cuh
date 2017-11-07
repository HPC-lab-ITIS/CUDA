__global__ void kernel(float *a)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	a[i] *= i;
}


int zero_copy(std::vector<float> &a, profiler &prof)
{
    auto n = a.size();
    size_t n_bytes = n * sizeof(float);
    float *a_dev = nullptr, *buffer = nullptr;

    cudaCheckError( cudaSetDevice(1) );
    cudaSetDeviceFlags(cudaDeviceMapHost); 
    cudaHostAlloc(&buffer, n_bytes, cudaHostAllocMapped);

    for (auto i = 0; i < n; i++)
        buffer[i] = a[i];

    cudaCheckError( cudaMalloc( &a_dev, n_bytes ) );
    cudaCheckError( cudaDeviceSynchronize() );   

    prof.tic("Zero copy");
    for(auto i = 0; i < 1000; ++i)
    {    
        kernel<<<13, 256>>>(buffer);
        cudaCheckError( cudaGetLastError() );
        cudaCheckError( cudaDeviceSynchronize() );
    }
    prof.toc("Zero copy");

    prof.tic("Standart copy");
    cudaThreadSynchronize();
    for(auto i = 0; i < 1000; ++i)
    {    
        cudaCheckError( cudaMemcpy(a_dev, a.data(), n_bytes, cudaMemcpyHostToDevice) );
        kernel<<<13, 256>>>(a_dev);
        cudaCheckError( cudaGetLastError() );
        cudaCheckError( cudaDeviceSynchronize() );        
        cudaCheckError( cudaMemcpy(a.data(), a_dev, n_bytes, cudaMemcpyDeviceToHost) );
    }
    cudaThreadSynchronize();
    prof.toc("Standart copy");

    cudaFree(a_dev);
    cudaFreeHost(buffer);

    return 0;
}
