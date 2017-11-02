#include <iostream>
#include <vector>
#include <random>
#include <functional>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>

__global__ void saxpy_kernel(float a, float *x, float *y, float *z)
{
	// Вычисляем глобальный индекс нити
	size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

	// Обработка соответствующих каждой нити данных
        z[idx] = a * x[idx] + y[idx];
}

int saxpy_wrapper(std::vector<float> &x, std::vector<float> &y, std::vector<float> &z, float a)
{
    int n = x.size();
    auto n_bytes = n * sizeof(float);
    float *x_dev = nullptr, *y_dev = nullptr, *z_dev = nullptr;

    //Выделить память на GPU для x_dev
    cudaError_t cuerr = cudaMalloc( (void**)&x_dev, n_bytes );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot allocate GPU memory for x_dev" << cudaGetErrorString(cuerr);
        return 1;
    }

    //Выделить память на GPU для н_dev
    cuerr = cudaMalloc( (void**)&y_dev, n_bytes );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot allocate GPU memory for y_dev" << cudaGetErrorString(cuerr);
        return 1;
    }

    //Выделить память на GPU для z_dev
    cuerr = cudaMalloc( (void**)&z_dev, n_bytes );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot allocate GPU memory for z_dev" << cudaGetErrorString(cuerr);
        return 1;
    }

    //Задать конфигурацию запуска блоков нитей и сетки блоков
    int block_size = 1024;
    int grid_size = n / block_size;

    //Скопировать входные данные из памяти CPU в память GPU.
    cuerr = cudaMemcpy(x_dev, x.data(), n_bytes, cudaMemcpyHostToDevice );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot copy data from x to x_dev" << cudaGetErrorString(cuerr);
        return 1;
    }

    //Скопировать входные данные из памяти CPU в память GPU.
    cuerr = cudaMemcpy(y_dev, y.data(), n_bytes, cudaMemcpyHostToDevice );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot copy data from y to y_dev" << cudaGetErrorString(cuerr);
        return 1;
    }

    //Создать события для замерения времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    // Вызвать ядро с заданной конфигурацией для обработки данных в цикле
    saxpy_kernel<<<grid_size, block_size>>>(a, x_dev, y_dev, z_dev);

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
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    //Вывести время исполнения в мс
    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop);
    std::cout << "Elapsed time gpu: " << gpu_time << " ms." << std::endl;

    // Скопировать результаты в память CPU.
    cuerr = cudaMemcpy(z.data(), z_dev, n_bytes, cudaMemcpyDeviceToHost );
    if (cuerr != cudaSuccess)
    {
        std::cout << "Cannot copy data from z_dev to z " << cudaGetErrorString(cuerr);
        return 1;
    }

    // Освободить выделенную память GPU.
    cudaFree(x_dev);
    cudaFree(y_dev);
    cudaFree(z_dev);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

int main(int argc, char *argv[])
{
    size_t n = 1 << 27;

    std::vector<float> x(n);
    std::vector<float> y(n);
    std::vector<float> z_cpu(n);
    std::vector<float> z_gpu(n);

    std::uniform_real_distribution<> distribution(0.0, 1.0);
    std::mt19937 engine; 
    auto generator = std::bind(distribution, engine);
    std::generate_n(x.begin(), n, generator);
    std::generate_n(y.begin(), n, generator);
    float a = distribution(engine);

    auto begin = std::chrono::steady_clock::now();
    for(auto i = 0; i < n; ++i)
        z_cpu[i] = a * x[i] + y[i];
    auto end = std::chrono::steady_clock::now();
    
    std::cout << "Elapsed time cpu: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms." << std::endl;

    saxpy_wrapper(x, y, z_gpu, a);

    for(auto i = 0; i < n; ++i)
        if( fabs( z_cpu[i] - z_gpu[i] ) > 1e-5)
        {
            std::cout << "Wrong calculation" << std::endl;
            return 1;
        }

    return 0;
}
