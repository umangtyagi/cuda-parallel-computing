#pragma once
#include <cuda_runtime.h>

class CudaTimer {
private:
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

public:
    CudaTimer() {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        cudaEventRecord(startEvent);
    }
    ~CudaTimer() {
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }
    void start() {
        cudaEventRecord(startEvent);
    }
    double elapsedTime() {
        float ms = 0;
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&ms, startEvent, stopEvent);
        return ms / 1000.0;
    }
};