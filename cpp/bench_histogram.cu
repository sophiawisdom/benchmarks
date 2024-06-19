#include "curand_kernel.h"
#include <assert.h>
#include <stdio.h>

constexpr int CACHE_SIZE = 128;
constexpr int MAX_BIN_SIZE = 4096;

enum Strategy {
    SHMEM,
    BALLOT
};

template<int bins>
__global__ void bench(int *outs, int items_per_thread, int threads) {
    curandState_t state;
    constexpr int seed = 134;
    curand_init(seed + threadIdx.x, 0, 0, &state);

    __shared__ int bin_values[MAX_BIN_SIZE];
    #pragma unroll 1
    for (int i = 0; i < MAX_BIN_SIZE; i++) {
        bin_values[i] = 0;
    }


    __syncthreads();
    unsigned long long diff = 0;

    for (int i = 0; i < 32; i++) {
        // precompute bins
        unsigned int rand_data[CACHE_SIZE];
        #pragma unroll CACHE_SIZE
        for (int i = 0; i < CACHE_SIZE; i++) {
            rand_data[i] = curand(&state)%MAX_BIN_SIZE;
        }
        unsigned int start;
        asm volatile("mov.u32 %0, %clock;" : "=r"(start));
        __syncwarp();

        for (int i = 0; i < (items_per_thread/CACHE_SIZE); i++) {
            #pragma unroll CACHE_SIZE
            for (int j = 0; j < CACHE_SIZE; j++) {
                atomicAdd(&bin_values[rand_data[j]], 1);
            }
        }

        __syncwarp();
        unsigned int end;
        asm volatile("mov.u32 %0, %clock;" : "=r"(end));
        diff += end - start;
    }

    if (threadIdx.x % 32 == 0) {
        outs[(blockIdx.x * blockDim.x + threadIdx.x)/32] = (int)diff;
    }
}

int bench_histogram(
    int * outs,
    int bins,
    int items_per_thread,
    int threads
) {
    assert(threads % 32 == 0);

    using kernel_ptr = void(*)(int*, int, int);
    kernel_ptr kernel = nullptr;
    #define ASSIGN_KERNEL(BINS) if (bins == BINS) {kernel = &bench<BINS>;}

    ASSIGN_KERNEL(1);
    ASSIGN_KERNEL(2);
    ASSIGN_KERNEL(4);
    ASSIGN_KERNEL(8);
    ASSIGN_KERNEL(16);
    ASSIGN_KERNEL(32);
    ASSIGN_KERNEL(64);
    ASSIGN_KERNEL(128);
    ASSIGN_KERNEL(256);

    assert(bins <= 256);


    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (kernel) {
        kernel<<<deviceProp.multiProcessorCount, threads>>>(outs, items_per_thread, threads);
        return 0;
    } else {
        printf("RUNNING NOTHING for bins=%d\n", bins);
        return 1;
    }
}