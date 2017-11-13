__global__ void matmult_shared(float* a, float* b, float* c, int n)
{
    //Подматрица A в разделяемой памяти
    __shared__ float a_s[block_size][block_size];
    //Подматрица B в разделяемой памяти
    __shared__ float b_s[block_size][block_size];

    //Вычисляемый элемент C’
    float sum = 0.;

    int i =  threadIdx.y + blockDim.y * blockIdx.y;
    int j = threadIdx.x + blockDim.x * blockIdx.x;

    //Цикл по подматрицам
    for (int m = 0; m < n / block_size; ++m)
    {
        // Загрузить по одному элементу из A и B в разделяемую память
        a_s[threadIdx.y][threadIdx.x] = a[i * n + (m * block_size + threadIdx.x)];
        b_s[threadIdx.y][threadIdx.x] = b[(m * block_size + threadIdx.y) * n + j];

        //Дождаться когда обе подматрицы будут полностью загружены
        __syncthreads();

        //Вычислить элемент произведения загруженных подматриц
        for (int k = 0; k < block_size; k++)
            sum += a_s[threadIdx.y][k] * b_s[k][threadIdx.x];

        //Дождаться пока все нити блока закончат
        __syncthreads();
    }

    //Сохранить результат в глобальной памяти
    c [i * n + j] = sum;
}


void matmult_shared(std::vector<float> &a, std::vector<float> &b, std::vector<float> &c, profiler &prof)
{
    auto n_bytes = n * n * sizeof(float);
    float *a_dev = nullptr, *b_dev = nullptr, *c_dev = nullptr;

    cudaCheckError( cudaMalloc(&a_dev, n_bytes) );
    cudaCheckError( cudaMalloc(&b_dev, n_bytes) );
    cudaCheckError( cudaMalloc(&c_dev, n_bytes) );

    cudaCheckError( cudaMemcpy (a_dev, a.data(), n_bytes, cudaMemcpyHostToDevice) );
    cudaCheckError( cudaMemcpy (b_dev, b.data(), n_bytes, cudaMemcpyHostToDevice) );

    dim3 block(block_size, block_size);
    dim3 grid(n / block_size, n / block_size);   

    prof.tic("mult shared");
    matmult_shared<<<grid, block>>>(a_dev, b_dev, c_dev, n);
    cudaCheckError( cudaGetLastError() );
    cudaCheckError( cudaDeviceSynchronize() );
    prof.toc("mult shared");
    
    cudaCheckError( cudaMemcpy (c.data(), c_dev, n_bytes, cudaMemcpyDeviceToHost) );

    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);
}
