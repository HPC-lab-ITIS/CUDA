int cudaCheckError(cudaError_t cuerr)
{
    if (cuerr != cudaSuccess)
    {
        std::cout << cudaGetErrorString(cuerr) << std::endl;
        return 1;
    }

    return 0;
}


int pinned_copy(std::vector<float> &a, profiler &prof)
{
    auto n = a.size();
    auto n_bytes = n * sizeof(float);
    float *a_dev = nullptr, *buffer = nullptr;

    cudaCheckError( cudaSetDevice(1) );

    cudaMallocHost( &buffer, n_bytes);

    for (auto i = 0; i < n; i++)
        buffer[i] = a[i];

    cudaCheckError( cudaMalloc( &a_dev, n_bytes ) );
    cudaCheckError( cudaDeviceSynchronize() );

    prof.tic("Pinned memory");
    for(auto i = 0; i < 1000; ++i)
    {    
        cudaCheckError( cudaMemcpy(a_dev, buffer, n_bytes, cudaMemcpyHostToDevice) );
        cudaCheckError( cudaMemcpy(buffer, a_dev, n_bytes, cudaMemcpyDeviceToHost) );
    }
    cudaCheckError( cudaDeviceSynchronize() );
    prof.toc("Pinned memory");

    prof.tic("Paged memory");
    cudaCheckError( cudaDeviceSynchronize() );
    for(auto i = 0; i < 1000; ++i)
    {    
        cudaCheckError( cudaMemcpy(a_dev, a.data(), n_bytes, cudaMemcpyHostToDevice) );
        cudaCheckError( cudaMemcpy(a.data(), a_dev, n_bytes, cudaMemcpyDeviceToHost) );
    }
    cudaCheckError( cudaDeviceSynchronize() );
    prof.toc("Paged memory");

    cudaFree(a_dev);
    cudaFreeHost(buffer);

    return 0;
}
