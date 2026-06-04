#include <cuda_runtime.h>
#include <iostream>

int main() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0) {
    std::cout << "No CUDA devices found!" << std::endl;
    return 1;
  }

  std::cout << "Found " << deviceCount << " CUDA device(s)\n" << std::endl;

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    std::cout << "=== Device " << i << ": " << prop.name << " ===" << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor
              << std::endl;
    std::cout << "Total Memory: " << prop.totalGlobalMem / (1024 * 1024)
              << " MB" << std::endl;
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max Threads/Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    std::cout << std::endl;
  }

  return 0;
}