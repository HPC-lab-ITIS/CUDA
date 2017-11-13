void matmult_cublas(std::vector<float> &a, std::vector<float> &b, std::vector<float> &c, profiler &prof)
{
    auto n_bytes = n * n * sizeof(float);
    float *a_dev = nullptr, *b_dev = nullptr, *c_dev = nullptr;

    cudaCheckError( cudaMalloc(&a_dev, n_bytes) );
    cudaCheckError( cudaMalloc(&b_dev, n_bytes) );
    cudaCheckError( cudaMalloc(&c_dev, n_bytes) );

    cudaCheckError( cudaMemcpy (a_dev, a.data(), n_bytes, cudaMemcpyHostToDevice) );
    cudaCheckError( cudaMemcpy (b_dev, b.data(), n_bytes, cudaMemcpyHostToDevice) );

    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    prof.tic("mult cublas");
    for(int i = 0; i < 2; ++i)
    {
        float alpha = 1, beta = 0;
        cublasSgemm_v2(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                &alpha, a_dev, n, b_dev, n, &beta, c_dev, n);
        prof.toc("mult cublas");

        cudaCheckError( cudaDeviceSynchronize() );
    }

    cudaCheckError( cudaMemcpy (c.data(), c_dev, n_bytes, cudaMemcpyDeviceToHost) );
    
    cublasDestroy_v2(handle);
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);
}
