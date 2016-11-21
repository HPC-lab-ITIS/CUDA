#include <iostream>
#include <vector>
#include "profiler.h"
#include <algorithm>

__global__ void matmult_naive(double* a, double* b, double* c, size_t n)
{
    double sum = 0.;
    // Смещение для a [i][0]
    int i_a = n * ( threadIdx.y +  blockDim.y * blockIdx.y );
    // Смещение для b [0][j]
    int i_b = threadIdx.x + blockDim.x * blockIdx.x;
    // Перемножить строку и столбец
    for (auto k = 0; k < n; k++)
        sum += a[i_a + k] * b[i_b + k * n];
    // Смещение для записываемого элемента
    int i_c = + ;
    // Сохранить результат в глобальной памяти
    c[(threadIdx.x + blockDim.x * blockIdx.x) + n * (threadIdx.y + blockDim.y * blockIdx.y)] = sum;
}


__global__ void matmult_block(double* a, double* b, double* c, int n)
{
    // Индекс начала первой подматрицы A обрабатываемой блоком.
    int a_begin = n * blockDim.y * blockIdx.y;
    int a_end = a_begin + n - 1;
    // Шаг перебора подматриц A.
    int a_step = blockDim.y;
    // Индекс первой подматрицы B обрабатываемой блоком
    int b_begin = blockDim.x * blockIdx.x;
    // Шаг перебора подматриц B
    int b_step = blockDim.x * n;
    // Вычисляемый элемент C’.
    double sum = 0.;

    const int block_size = 16;
    // Цикл по всем подматрицам
    for (int i_a = a_begin, i_b = b_begin; i_a <= a_end; i_a += a_step, i_b += b_step)
    {
        // Очередная подматрица A в разделяемой памяти.
        __shared__ double a_s[block_size][block_size];
        // Очередная подматрица B в разделяемой памяти.
        __shared__ double b_s[block_size][block_size];
        // Загрузить по одному элементу из A и B в разделяемую память.
        a_s[threadIdx.y][threadIdx.x] = a[i_a + threadIdx.x + n * threadIdx.y];
        b_s[threadIdx.y][threadIdx.x] = b[i_b + threadIdx.x + n * threadIdx.y];
        // Дождаться когда обе подматрицы будут полностью загружены.
        __syncthreads();
        // Вычислить элемент произведения загруженных подматриц.
        for (int k = 0; k < blockDim.x; k++)
            sum += a_s[threadIdx.y][k] * b_s[k][threadIdx.x];
        // Дождаться пока все остальные нити блока закончат вычислять
        // свои элементы.
        __syncthreads();
    }
    // Записать результат.
    int i_c = n * blockDim.y * blockIdx.y + blockDim.x * blockIdx.x;
    c [i_c + threadIdx.x + n * threadIdx.y] = sum;
}

int gpu_mult(std::vector<double> &a, std::vector<double> &b,
        std::vector<double> &c, size_t n)
{
    auto n_bytes = n * n * sizeof(double);
    double *a_dev = nullptr, *b_dev = nullptr, *c_dev = nullptr;
    profiler prof;

    cudaError_t cuerr = cudaMalloc ( (void**)&a_dev, n_bytes );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot allocate GPU memory for a_dev" << cudaGetErrorString(cuerr);
        return 1;
    }

    cuerr = cudaMalloc ( (void**)&b_dev, n_bytes );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot allocate GPU memory for b_dev" << cudaGetErrorString(cuerr);
        return 1;
    }

    cuerr = cudaMalloc ( (void**)&c_dev, n_bytes );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot allocate GPU memory for c_dev" << cudaGetErrorString(cuerr);
        return 1;
    }

    cuerr = cudaMemcpy ( a_dev, a.data(), n_bytes, cudaMemcpyHostToDevice );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot copy data from a to a_dev" << cudaGetErrorString(cuerr);
        return 1;
    }

    cuerr = cudaMemcpy ( b_dev, b.data(), n_bytes, cudaMemcpyHostToDevice );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot copy data from a to b_dev" << cudaGetErrorString(cuerr);
        return 1;
    }

    dim3 block_size(16,16);
    dim3 grid_size(208,208);

    cudaThreadSynchronize();
    prof.tic("Naive mult");
    matmult_naive<<<grid_size, block_size>>>(a_dev, b_dev, c_dev, n);
    cudaThreadSynchronize();
    prof.toc("Naive mult");

    cudaThreadSynchronize();
    prof.tic("Block mult");
    matmult_block<<<grid_size, block_size>>>(a_dev, b_dev, c_dev, n);
    cudaThreadSynchronize();
    prof.toc("Block mult");

    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot launch CUDA kernel " << cudaGetErrorString(cuerr);
        return 1;
    }

    // Ожидать завершения работы ядра.
    cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot synchronize CUDA kernel " << cudaGetErrorString(cuerr);
        return 1;
    }

    // Скопировать результаты в память CPU.
    cuerr = cudaMemcpy ( c.data(), c_dev, n_bytes, cudaMemcpyDeviceToHost );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot copy data from c_dev to c " << cudaGetErrorString(cuerr);
        return 1;
    }

    // Освободить выделенную память GPU.
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);

    prof.report();

    return 0;	
}

int main()
{
    const auto n = 208*32;
    std::vector<double> a(n*n,0.);
    std::vector<double> b(n*n,0.);
    std::vector<double> c(n*n,0.);
    std::vector<double> c_host(n*n,0.);
    profiler prof;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::mt19937 engine; // Mersenne twister MT19937
    auto generator = std::bind(distribution, engine);
    std::generate_n(a.begin(), n, generator);
    std::generate_n(b.begin(), n, generator);

    gpu_mult(a, b, c, n);
    
/*    prof.tic("seq_mult");
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < n; ++j)
            for(int k = 0; k < n; ++k)
                c_host[i*n + j] += a[i*n + k] * b[k*n + j];
    prof.toc("seq_mult");
*/
    prof.report();

    for(auto i = 0; i < c.size(); ++i)
        if(fabs(c_host[i] - c[i])>1e-5)
        {
            std::cout << "fail" << std::endl;
            return 1;
        }

    return 0;
}
