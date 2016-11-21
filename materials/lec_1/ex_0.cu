#include <iostream>

int main()
{
    cudaDeviceProp prop;
    int count = 0;
    cudaGetDeviceCount(&count);
    for(int i = 0; i < count; ++i)
    {
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Information for device #" << i << std::endl;
        std::cout << "Name " << prop.name << std::endl;
        std::cout << "Compute capability " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Total memory " << prop.totalGlobalMem / (1024*1024*1024) << "GB" << std::endl;
        std::cout << "Multiprocessor count " << prop.multiProcessorCount << std::endl;
        std::cout << "Threads in warp " << prop.warpSize << std::endl;
        std::cout << "Max threads per block " << prop.maxThreadsPerBlock << std::endl;
    }

    return 0;
}
