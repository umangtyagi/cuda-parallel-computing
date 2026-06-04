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

// NOTE : too slow for initialization; used curandGenerator_t instead
// Kernel for random initialization on GPU
// __global__ void init_random_kernel(double *a, double *b, int n,
//                                    unsigned long seed) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < n) {
//     curandState state;
//     curand_init(seed, idx, 0, &state); // each thread gets unique sequence
//     a[idx] = curand_uniform_double(&state);
//     b[idx] = curand_uniform_double(&state);
//   }
// }

// Fill vectors a and b with random doubles on the CPU
void fill_vectors_random_cpu(std::vector<double> &a, std::vector<double> &b,
                             long n) {
  std::random_device rd;
  std::mt19937 gen(42); // fixed seed for reproducability

  // doubles between 0.0 and 100.0
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  // fill array
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

  // memory allocation on device
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_result, bytes);

  // copy host -> device
  ct.start();
  cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);
  double h2d = ct.elapsedTime();
  std::cout << "GPU H->D copy:  " << h2d << "s\n";
  std::cout << "H->D bandwidth: " << (2.0 * bytes / 1e9) / h2d << " GB/s\n";

  // kernel
  int numBlocks = (N + blockSize - 1) / blockSize;
  ct.start();
  vector_add_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_result, N);
  std::cout << "GPU kernel:     " << ct.elapsedTime() << "s\n";

  // copy device -> host
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
  std::cout << "\n--- CPU init ---\n";
  StopWatch sw;

  // Memory allocation
  sw.start();
  std::vector<double> arr1(n);
  std::vector<double> arr2(n);
  std::vector<double> result_cpu(n);
  std::vector<double> result_gpu(n);
  std::cout << "CPU alloc:       " << sw.elapsedTime() << "s\n";

  // CPU initialization
  sw.start();
  fill_vectors_random_cpu(arr1, arr2, n);
  std::cout << "CPU init:        " << sw.elapsedTime() << "s\n";

  // CPU addition
  sw.start();
  add_vectors_cpu(arr1, arr2, result_cpu, n);
  std::cout << "CPU addition:    " << sw.elapsedTime() << "s\n";

  // GPU addition
  int blockSize = 1024;
  add_vectors_gpu(arr1, arr2, result_gpu, blockSize);

  compare_results(result_cpu, result_gpu);

  std::cout << "\n--- GPU init ---\n";

  // Memory allocation
  double *d_a, *d_b, *d_result;
  size_t bytes = n * sizeof(double);
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_result, bytes);

  // GPU initialization
  fill_vectors_random_gpu(d_a, d_b, n);

  // GPU addition
  CudaTimer ct;
  int numBlocks = (n + blockSize - 1) / blockSize;
  ct.start();
  vector_add_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_result, n);
  std::cout << "GPU kernel:      " << ct.elapsedTime() << "s\n";

  // D -> H
  ct.start();
  cudaMemcpy(arr1.data(), d_a, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(arr2.data(), d_b, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(result_gpu.data(), d_result, bytes, cudaMemcpyDeviceToHost);
  double d2h = ct.elapsedTime();
  std::cout << "GPU D->H copy:   " << d2h << "s\n";
  std::cout << "D->H bandwidth:  " << (3.0 * bytes / 1e9) / d2h << " GB/s\n";

  // CPU addition
  sw.start();
  add_vectors_cpu(arr1, arr2, result_cpu, n);
  std::cout << "CPU addition:    " << sw.elapsedTime() << "s\n";

  compare_results(result_cpu, result_gpu);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);
}