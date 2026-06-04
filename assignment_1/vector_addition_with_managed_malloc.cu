#include "../include/CudaTimer.h"
#include "../include/StopWatch.h"
#include <curand.h>
#include <curand_kernel.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// Kernel for vector addition on GPU
__global__ void vector_add_kernel(const double *a, const double *b,
                                  double *result, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    result[idx] = a[idx] + b[idx];
  }
}

// Fill vectors a and b with random doubles on the CPU
void fill_vectors_random_cpu(std::vector<double> &a, std::vector<double> &b,
                             long n) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (long i = 0; i < n; i++) {
    a[i] = dist(gen);
    b[i] = dist(gen);
  }
}

void fill_vectors_random_gpu(double *d_a, double *d_b, long n) {
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 42);

  CudaTimer ct;
  ct.start();
  curandGenerateUniformDouble(gen, d_a, n);
  curandGenerateUniformDouble(gen, d_b, n);
  std::cout << "GPU init:        " << ct.elapsedTime() << "s\n";

  curandDestroyGenerator(gen);
}

void add_vectors_cpu(const std::vector<double> &a, const std::vector<double> &b,
                     std::vector<double> &result, long n) {
  for (long i = 0; i < n; i++) {
    result[i] = a[i] + b[i];
  }
}

void add_vectors_gpu(const std::vector<double> &a, const std::vector<double> &b,
                     std::vector<double> &result, const int blockSize) {
  double *d_a, *d_b, *d_result;
  long N = a.size();
  size_t bytes = N * sizeof(double);

  CudaTimer ct;

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_result, bytes);

  ct.start();
  cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);
  double h2d = ct.elapsedTime();
  std::cout << "GPU H->D copy:  " << h2d << "s\n";
  std::cout << "H->D bandwidth: " << (2.0 * bytes / 1e9) / h2d << " GB/s\n";

  int numBlocks = (N + blockSize - 1) / blockSize;
  ct.start();
  vector_add_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_result, N);
  std::cout << "GPU kernel:     " << ct.elapsedTime() << "s\n";

  ct.start();
  cudaMemcpy(result.data(), d_result, bytes, cudaMemcpyDeviceToHost);
  double d2h = ct.elapsedTime();
  std::cout << "GPU D->H copy:  " << d2h << "s\n";
  std::cout << "D->H bandwidth: " << (bytes / 1e9) / d2h << " GB/s\n";

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);
}

bool compare_results(const std::vector<double> &r1,
                     const std::vector<double> &r2) {
  for (size_t i = 0; i < r1.size(); i++) {
    if (std::abs(r1[i] - r2[i]) > 1e-5) {
      std::cout << "Results do not match at idx : " << i << std::endl;
      return false;
    }
  }
  std::cout << "Results match!" << std::endl;
  return true;
}

int main() {

  long n = 9000000;
  std::cout << std::fixed << std::setprecision(6);
  int blockSize = 1024;
  StopWatch sw;

  // =========================================================
  // PATH 1: CPU init, manual GPU memory
  // =========================================================
  std::cout << "\n--- CPU init / manual memory ---\n";

  sw.start();
  std::vector<double> arr1(n);
  std::vector<double> arr2(n);
  std::vector<double> result_cpu(n);
  std::vector<double> result_gpu(n);
  std::cout << "CPU alloc:       " << sw.elapsedTime() << "s\n";

  sw.start();
  fill_vectors_random_cpu(arr1, arr2, n);
  std::cout << "CPU init:        " << sw.elapsedTime() << "s\n";

  sw.start();
  add_vectors_cpu(arr1, arr2, result_cpu, n);
  std::cout << "CPU addition:    " << sw.elapsedTime() << "s\n";

  add_vectors_gpu(arr1, arr2, result_gpu, blockSize);
  compare_results(result_cpu, result_gpu);

  // =========================================================
  // PATH 2: GPU init, managed memory
  // =========================================================
  std::cout << "\n--- GPU init / managed memory ---\n";

  double *d_a, *d_b, *d_result;
  size_t bytes = n * sizeof(double);

  sw.start();
  cudaMallocManaged(&d_a, bytes);
  cudaMallocManaged(&d_b, bytes);
  cudaMallocManaged(&d_result, bytes);
  std::cout << "GPU alloc:       " << sw.elapsedTime() << "s\n";

  fill_vectors_random_gpu(d_a, d_b, n);

  CudaTimer ct;
  int numBlocks = (n + blockSize - 1) / blockSize;
  ct.start();
  vector_add_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_result, n);
  std::cout << "GPU kernel:      " << ct.elapsedTime() << "s\n";

  cudaDeviceSynchronize();

  sw.start();
  for (long i = 0; i < n; i++)
    result_cpu[i] = d_a[i] + d_b[i];
  std::cout << "CPU addition:    " << sw.elapsedTime() << "s\n";

  for (long i = 0; i < n; i++) {
    if (std::abs(result_cpu[i] - d_result[i]) > 1e-5) {
      std::cout << "Results do not match at idx : " << i << std::endl;
      break;
    }
  }
  std::cout << "Results match!\n";

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);
}