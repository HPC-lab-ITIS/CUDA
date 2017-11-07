__constant__ float turn_matrix[4];


__global__ void turn_kernel(float *x, float *y)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float tmp_x = x[idx] * turn_matrix[0] + y[idx] * turn_matrix[1];
  float tmp_y = x[idx] * turn_matrix[2] + y[idx] * turn_matrix[3];

  x[idx] = tmp_x;
  y[idx] = tmp_y;
}


__global__ void turn_kernel(float *x, float *y, float *tmp)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float tmp_x = x[idx] * tmp[0] + y[idx] * tmp[1];
  float tmp_y = x[idx] * tmp[2] + y[idx] * tmp[3];

  x[idx] = tmp_x;
  y[idx] = tmp_y;
}


int cudaCheckError(cudaError_t cuerr)
{
    if (cuerr != cudaSuccess)
    {
        std::cout << cudaGetErrorString(cuerr) << std::endl;
        return 1;
    }

    return 0;
}


int turn(std::vector<double> &x, std::vector<double> &y, const float angle, profiler &prof)
{
    auto n = x.size();
    auto n_bytes = n * sizeof(float);
    float *x_dev = nullptr, *y_dev = nullptr, *tmp_dev = nullptr;
    float tmp[4];

    tmp[0] = cos(angle);
    tmp[1] = -sin(angle);
    tmp[2] = sin(angle);
    tmp[3] = cos(angle);

    cudaCheckError( cudaSetDevice(1) );

    cudaCheckError( cudaMalloc( &x_dev, n_bytes ) );

    cudaCheckError( cudaMalloc( &y_dev, n_bytes ) );

    cudaCheckError( cudaMalloc( &tmp_dev, sizeof(float) * 4 ) );
    
    cudaCheckError( cudaMemcpy(x_dev, x.data(), n_bytes, cudaMemcpyHostToDevice) );
    
    cudaCheckError( cudaMemcpy(y_dev, y.data(), n_bytes, cudaMemcpyHostToDevice) );
    
    cudaCheckError( cudaMemcpyToSymbol(turn_matrix, tmp, sizeof(float) * 4) );

    cudaCheckError( cudaMemcpy(tmp_dev, tmp, sizeof(float) * 4, cudaMemcpyHostToDevice) );
    
    int threads = 1024;
    int blocks = n / threads;
    
    prof.tic("gpu turn global memory");
    for(int i = 0; i < 10; ++i)
    {
        turn_kernel<<<blocks, threads>>>(x_dev, y_dev, tmp_dev);
        cudaCheckError( cudaGetLastError() );
        cudaCheckError( cudaDeviceSynchronize() );
    }
    prof.toc("gpu turn global memory");
    
    prof.tic("gpu turn constant memory");
    for(int i = 0; i < 10; ++i)
    {
        turn_kernel<<<blocks, threads>>>(x_dev, y_dev);
        cudaCheckError( cudaGetLastError() );
        cudaCheckError( cudaDeviceSynchronize() );
    }
    prof.toc("gpu turn constant memory");

    cudaFree(x_dev);
    cudaFree(y_dev);
    cudaFree(tmp_dev);

    return 0;
}
