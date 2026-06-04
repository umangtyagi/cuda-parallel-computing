#include "../include/CudaTimer.h"
#include "../include/StopWatch.h"
#include <algorithm>
#include <cfloat>
#include <curand.h>
#include <curand_kernel.h>
#include <iomanip>
#include <iostream>
#include <vector>

struct MaxPair {
  double value;
  long index;
};

// pass 1: reads raw doubles, outputs MaxPair
__global__ void max_reduction_kernel(const double *arr, MaxPair *block_results,
                                     long n) {
  extern __shared__ MaxPair sdata[];
  int tid = threadIdx.x;
  long idx = (long)blockIdx.x * blockDim.x + tid;

  sdata[tid] = (idx < n) ? MaxPair{arr[idx], idx} : MaxPair{-DBL_MAX, -1};
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && sdata[tid + s].value > sdata[tid].value)
      sdata[tid] = sdata[tid + s];
    __syncthreads();
  }

  if (tid == 0)
    block_results[blockIdx.x] = sdata[0];
}

// pass 2: reads MaxPair, outputs MaxPair
__global__ void maxpair_reduction_kernel(const MaxPair *arr,
                                         MaxPair *block_results, long n) {
  extern __shared__ MaxPair sdata[];
  int tid = threadIdx.x;
  long idx = (long)blockIdx.x * blockDim.x + tid;

  sdata[tid] = (idx < n) ? arr[idx] : MaxPair{-DBL_MAX, -1};
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && sdata[tid + s].value > sdata[tid].value)
      sdata[tid] = sdata[tid + s];
    __syncthreads();
  }

  if (tid == 0)
    block_results[blockIdx.x] = sdata[0];
}

// GPU random init
void fill_vectors_random_gpu(double *d_arr, long n) {
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 42);

  CudaTimer ct;
  ct.start();
  curandGenerateUniformDouble(gen, d_arr, n);
  std::cout << "GPU init:        " << ct.elapsedTime() << "s\n";

  curandDestroyGenerator(gen);
}

std::pair<double, long> find_max_cpu(const std::vector<double> &arr) {

  double max_val = arr[0];
  long max_idx = 0;

  for (long i = 0; i < (long)arr.size(); i++) {
    if (arr[i] > max_val) {
      max_val = arr[i];
      max_idx = i;
    }
  }
  return {max_val, max_idx};
}

int main() {

  long n = 90000000;
  int blockSize = 1024;

  std::cout << std::fixed << std::setprecision(6);
  StopWatch sw;

  // Memory allocation
  std::vector<double> arr(n);
  double *d_arr;
  size_t bytes = n * sizeof(double);
  cudaMalloc(&d_arr, bytes);

  // GPU init
  fill_vectors_random_gpu(d_arr, n);

  // GPU Pass 1 -> block maxes
  int numBlocks = (n + blockSize - 1) / blockSize;
  MaxPair *d_block_results;
  cudaMalloc(&d_block_results, numBlocks * sizeof(MaxPair));
  size_t sharedMem = blockSize * sizeof(MaxPair);

  CudaTimer ct;
  ct.start();
  max_reduction_kernel<<<numBlocks, blockSize, sharedMem>>>(d_arr,
                                                            d_block_results, n);
  double pass1_time = ct.elapsedTime();
  std::cout << "\n--- GPU reduction pass 1 ---\n";
  std::cout << "GPU pass 1:      " << pass1_time << "s\n";

  // GPU pass 2: block maxes -> 1 value (pure GPU)
  int numBlocks2 = (numBlocks + blockSize - 1) / blockSize;
  MaxPair *d_final;

  cudaMalloc(&d_final, numBlocks2 * sizeof(MaxPair));

  ct.start();

  maxpair_reduction_kernel<<<numBlocks2, blockSize, sharedMem>>>(
      d_block_results, d_final, numBlocks);

  double pass2_time = ct.elapsedTime();

  std::vector<MaxPair> final_result(numBlocks2);

  cudaMemcpy(final_result.data(), d_final, numBlocks2 * sizeof(MaxPair),
             cudaMemcpyDeviceToHost);

  MaxPair gpu_result = final_result[0];

  for (int i = 1; i < numBlocks2; i++) {
    if (final_result[i].value > gpu_result.value)
      gpu_result = final_result[i];
  }

  std::cout << "\n--- Pure GPU result ---\n";
  std::cout << "GPU pass 2:      " << pass2_time << "s\n";
  std::cout << "GPU total:       " << pass1_time + pass2_time << "s\n";
  std::cout << "GPU max value:   " << gpu_result.value << "\n";
  std::cout << "GPU max index:   " << gpu_result.index << "\n";
  cudaFree(d_final);

  // GPU + CPU final pass: copy block_max to host, scan on CPU
  std::vector<MaxPair> block_max(numBlocks);

  ct.start();

  cudaMemcpy(block_max.data(), d_block_results, numBlocks * sizeof(MaxPair),
             cudaMemcpyDeviceToHost);

  cudaFree(d_block_results);

  double block_d2h = ct.elapsedTime();

  sw.start();

  MaxPair gpu_cpu_result = block_max[0];

  for (int i = 1; i < numBlocks; i++) {
    if (block_max[i].value > gpu_cpu_result.value)
      gpu_cpu_result = block_max[i];
  }
  double cpu_scan_time = sw.elapsedTime();
  std::cout << "\n--- GPU + CPU final pass ---\n";
  std::cout << "GPU pass 1:      " << pass1_time << "s (reused)\n";
  std::cout << "Block D->H copy: " << block_d2h << "s\n";
  std::cout << "CPU final pass:  " << cpu_scan_time << "s\n";
  std::cout << "GPU+CPU total:   " << pass1_time + block_d2h + cpu_scan_time
            << "s\n";
  std::cout << "GPU+CPU max:     " << gpu_cpu_result.value << "\n";

  std::cout << "GPU+CPU index:   " << gpu_cpu_result.index << "\n";
  // D -> H
  ct.start();
  std::cout << "\n--- Pure CPU ---\n";
  cudaMemcpy(arr.data(), d_arr, bytes, cudaMemcpyDeviceToHost);
  cudaFree(d_arr);
  double d2h = ct.elapsedTime();
  std::cout << "GPU D->H copy:   " << d2h << "s\n";
  std::cout << "D->H bandwidth:  " << (bytes / 1e9) / d2h << " GB/s\n";

  // Find max on CPU (Pure CPU)
  sw.start();
  auto [cpu_max_val, cpu_max_idx] = find_max_cpu(arr);
  std::cout << "CPU find max:    " << sw.elapsedTime() << "s\n";
  std::cout << "CPU max value:   " << cpu_max_val << "\n";
  std::cout << "CPU max index:   " << cpu_max_idx << "\n";

  // Comparison
  std::cout << "\n--- Comparison ---\n";
  std::cout << "Pure GPU max:    " << gpu_result.value << "\n";

  std::cout << "Pure GPU index:  " << gpu_result.index << "\n";
  std::cout << "GPU+CPU max:     " << gpu_cpu_result.value << "\n";

  std::cout << "GPU+CPU index:   " << gpu_cpu_result.index << "\n";
  std::cout << "Pure CPU max:    " << cpu_max_val << "\n";
  std::cout << "CPU max index:   " << cpu_max_idx << "\n";

  if (std::abs(gpu_result.value - cpu_max_val) < 1e-10 &&
      std::abs(gpu_cpu_result.value - cpu_max_val) < 1e-10 &&
      gpu_result.index == cpu_max_idx && gpu_cpu_result.index == cpu_max_idx)
    std::cout << "All results match!\n";
  else
    std::cout << "Results do not match!\n";
}