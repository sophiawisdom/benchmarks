#include "stdio.h"

#include "assert.h"

#include "curand_kernel.h"

using namespace std;

#include <stdexcept>
#include <string>

// Macro to check CUDA errors
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(error) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while (0)

enum L2Operation {
    Scan,
};

enum TickTock {
    Tick,
    Tock
};

// PAIN
// https://x.com/cis_female/status/1795481684487639175
__device__ __forceinline__  int load_volatile(int *ptr) {
    int out;
    asm("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(out) : "l"(ptr));
    return out;
}
__device__ __forceinline__  int store_volatile(int *ptr, int value) {
    asm volatile("st.relaxed.gpu.global.s32 [%0], %1;" :: "l"(ptr), "r"(value));
}
__device__ __forceinline__ void no_ticktok_sync(int *scratchpad) {
    if (threadIdx.x == 0) {
        while (load_volatile(&scratchpad[1])) {} // wait for previous sync to complete
        int res = atomicAdd(&scratchpad[0], 1); // start counting # of arrived blocks
        if (res == (gridDim.x-1)) { // if we're the last block
            store_volatile(&scratchpad[1], 1); // mark completion
        }
        while (!load_volatile(&scratchpad[1])) {} // wait until completion
        if (atomicSub(&scratchpad[0], 1) == 1) { // check if we're the last block
            store_volatile(&scratchpad[1], 0); // if we are, reset completion marker
        }
    }
    __syncthreads();
}


template<TickTock tt>
__device__ __forceinline__ void global_sync(int *scratchpad) {
    __syncthreads();
    if (threadIdx.x == 0) {
        if constexpr (tt == Tick) {
            // scratchpad[1]?
            while (load_volatile(&scratchpad[4])) {}
            int res = atomicAdd(&scratchpad[0], 1);
            if (res == (gridDim.x-1)) {
                store_volatile(&scratchpad[1], 1);
            }
            while (!load_volatile(&scratchpad[1])) {}
            if (atomicSub(&scratchpad[0], 1) == 1) {
                store_volatile(&scratchpad[0], 0);
                store_volatile(&scratchpad[1], 0);
                store_volatile(&scratchpad[2], 0);
            }
        } else {
            while (load_volatile(&scratchpad[1])) {}
            int res = atomicAdd(&scratchpad[3], 1);
            if (res == (gridDim.x-1)) {
                store_volatile(&scratchpad[4], 1);
            }
            while (!load_volatile(&scratchpad[4])) {}
            if (atomicSub(&scratchpad[3], 1) == 1) {
                store_volatile(&scratchpad[3], 0);
                store_volatile(&scratchpad[4], 0);
                store_volatile(&scratchpad[5], 0);
            }
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void trash_l2(int *l2_scratchpad, curandState_t state) {
    // trash L2 with scratchpad
    constexpr int l2_scratchpad_size = 100*1024*1024;
    constexpr int u64_l2 = l2_scratchpad_size/8;
    constexpr int bytes_per_iter = 1024; // 8 bytes for u64 * 128
    constexpr int num_iters = l2_scratchpad_size/bytes_per_iter;
    void *void_scratchpad = (void *)l2_scratchpad;
    for (int i = 0; i < 1000; i++) {
        int read_idx = (curand(&state)) % u64_l2;
        int write_idx = (curand(&state)) % u64_l2;
        unsigned long long value;
        asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(value) : "l"(void_scratchpad+write_idx*8));
        asm volatile("st.cg.global.u64 [%0], %1;" :: "l"(void_scratchpad+read_idx*8), "l"(value ^ curand(&state)));
    }
}

template<L2Operation op>
__global__ void l2_bench_gpu(int *scratchpad, int *outs, int *l2_scratchpad) {
    curandState_t state;
    curand_init(threadIdx.x, 0, 0, &state);
    int smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    int nsmid;
    asm("mov.u32 %0, %nsmid;" : "=r"(nsmid));

    assert(nsmid == gridDim.x);

    __shared__ unsigned int x[128];
    // assert(blockDim.x == 128);

    for (int k = 0; k < 1; k++) {

    no_ticktok_sync(scratchpad);
    // if (threadIdx.x == 0) {printf("block %d exited first sync %d %d %d %d %d %d\n", blockIdx.x, load_volatile(&scratchpad[0]), load_volatile(&scratchpad[1]), load_volatile(&scratchpad[2]), load_volatile(&scratchpad[3]), load_volatile(&scratchpad[4]), load_volatile(&scratchpad[5]));}
    no_ticktok_sync(scratchpad);
    // if (threadIdx.x == 0) {printf("block %d exited second sync %d %d %d %d %d %d\n", blockIdx.x, load_volatile(&scratchpad[0]), load_volatile(&scratchpad[1]), load_volatile(&scratchpad[2]), load_volatile(&scratchpad[3]), load_volatile(&scratchpad[4]), load_volatile(&scratchpad[5]));}

    for (int other_sm = 0; other_sm < gridDim.x; other_sm++) {
        // if (blockIdx.x == 0 && threadIdx.x == 0) {printf("smid %d at other %d\n", smid, other_sm);}
        int tidx = (threadIdx.x%32);
        void *base_scratchpad = ((void *)&scratchpad[128]); // per-sm cacheline
        void *new_scratchpad = base_scratchpad+8192*smid; // per-sm cacheline
        void *other_sm_scratchpad = base_scratchpad+8192*other_sm+tidx*4;

        for (int look_sm = 0; look_sm < gridDim.x; look_sm++) {
            trash_l2(l2_scratchpad, state);
            asm volatile("st.cg.global.u32 [%0], %1;" :: "l"(new_scratchpad+((threadIdx.x)%32)*4), "r"(curand(&state)));

            // if (blockIdx.x == 0 && threadIdx.x == 0) {printf("smid %d at other %d look %d\n", smid, other_sm, look_sm);}

            // establish DOMINANCE over an L2 cache line
            for (int i = 0; i < 1024; i++) {
                unsigned int value;
                int rot_tidx = ((threadIdx.x+16)%32);
                asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(new_scratchpad+tidx*4));
            }

            // if (blockIdx.x == 0 && threadIdx.x == 0) {printf("smid %d entered global sync %d/%d\n", smid, other_sm, look_sm);}
            __nanosleep(50000);
            no_ticktok_sync(scratchpad);
            // if (blockIdx.x == 0 && threadIdx.x == 0) {printf("smid %d exited global sync %d/%d\n", smid, other_sm, look_sm);}

            if (smid == look_sm) {
                unsigned long long start;
                asm volatile("mov.u64 %0, %clock64;" : "=l"(start));

                unsigned int value;
                asm volatile("ld.global.u32 %0, [%1];" : "=r"(value) : "l"(other_sm_scratchpad));

                unsigned int sum;
                atomicAdd(&x[threadIdx.x], value); // need to make sure this value is used somehow...

                unsigned long long end;
                asm volatile("mov.u64 %0, %clock64;" : "=l"(end));

                if (threadIdx.x % 32 == 0) {
                    unsigned int small_start = start;
                    unsigned int small_end = end;

                    int diff = (small_end - small_start);
                    int idx = smid * gridDim.x * 4 + other_sm * 4 + threadIdx.x/32;

                    outs[idx] = diff;
                }
                __syncthreads();
            }

            __nanosleep(50000);
            no_ticktok_sync(scratchpad);
        }
    }
    }
}



float *bench_l2(
    int *outs, // pinned memory of size blocks * threads * iterations / 32
    int op_arg,
    int blocks,
    int threads,
    int iterations
) {
    L2Operation op = (L2Operation)op_arg;
    using kernel_ptr = void(*)(int*, int*, int*);
    kernel_ptr kernel = nullptr;

    #define ASSIGN_KERNEL(OP) if (op == OP) {kernel = &l2_bench_gpu<OP>;}

    ASSIGN_KERNEL(Scan);

    if (!kernel) {
        printf("RUNNING NOTHING for op=%d\n", op);
        return nullptr;
    }

    int *scratchpad;
    int scratchpad_size = 1024*128;
    CUDA_CHECK(cudaMalloc(&scratchpad, sizeof(int) * scratchpad_size));

    int *l2_scratchpad;
    int l2_scratchpad_size = 100*1024*1024; // 100MB
    CUDA_CHECK(cudaMalloc(&l2_scratchpad, l2_scratchpad_size));

    int *gpu_outs;
    // assert(blocks == 128);
    int out_size = blocks * blocks * 4; // per iteration
    CUDA_CHECK(cudaMalloc(&gpu_outs, sizeof(int) * out_size));
    CUDA_CHECK(cudaMemset(gpu_outs, 0, sizeof(int) * out_size));

    constexpr int nodes_per_iteration = 5;
    cudaGraphNode_t *nodes = (cudaGraphNode_t *)malloc(sizeof(cudaGraphNode_t *) * iterations * nodes_per_iteration);
    cudaEvent_t *starts = (cudaEvent_t *)malloc(sizeof(cudaEvent_t *) * iterations);
    cudaEvent_t *ends = (cudaEvent_t *)malloc(sizeof(cudaEvent_t *) * iterations);
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaEventCreate(&starts[i]));
        CUDA_CHECK(cudaEventCreate(&ends[i]));
    }

    cudaGraph_t graph;
    CUDA_CHECK(cudaGraphCreate(&graph, 0));

    printf("about to create graph!\n");
    void* benchArgs[3] = { (int*)&scratchpad, (int*)&gpu_outs, (int *)&l2_scratchpad };
    cudaKernelNodeParams benchParams = {0};
    benchParams.func = (void*)kernel;
    // TODO: change back
    benchParams.gridDim = dim3(blocks, 1, 1);
    benchParams.blockDim = dim3(threads, 1, 1);
    benchParams.kernelParams = benchArgs;
    benchParams.sharedMemBytes = 4096;
    benchParams.extra = NULL;

    cudaMemsetParams memsetParams = {0};
    memsetParams.dst = scratchpad;
    memsetParams.elementSize = sizeof(int);
    memsetParams.height = 1;
    memsetParams.width = scratchpad_size;
    memsetParams.value = 0;

    printf("about to create nodes!\n");
    for (int iteration = 0; iteration < iterations; iteration++) {
        int idx = iteration*nodes_per_iteration;
        CUDA_CHECK(cudaGraphAddMemsetNode(     &nodes[idx+0], graph, iteration == 0 ? nullptr : &nodes[idx-3], iteration == 0 ? 0 : 3, &memsetParams));
        CUDA_CHECK(cudaGraphAddEventRecordNode(&nodes[idx+1], graph, iteration == 0 ? &nodes[idx] : &nodes[idx-2], iteration == 0 ? 1 : 3, starts[iteration]));
        CUDA_CHECK(cudaGraphAddKernelNode(     &nodes[idx+2], graph, iteration == 0 ? &nodes[idx] : &nodes[idx-3], iteration == 0 ? 2 : 4, &benchParams));
        CUDA_CHECK(cudaGraphAddEventRecordNode(&nodes[idx+3], graph, &nodes[idx+2], 1, ends[iteration]));
        CUDA_CHECK(cudaGraphAddMemcpyNode1D(   &nodes[idx+4], graph, &nodes[idx+2], 1, &outs[iteration * out_size], gpu_outs, sizeof(int) * out_size, cudaMemcpyDeviceToHost));
    }
    printf("about to launch graph!\n");

    cudaGraphExec_t graph_exec;
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
    CUDA_CHECK(cudaGraphLaunch(graph_exec, 0));

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(scratchpad));
    CUDA_CHECK(cudaFree(gpu_outs));

    float *diffs = (float *)malloc(sizeof(float) * iterations);
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaEventElapsedTime(&diffs[i], starts[i], ends[i]));
    }

    // cudaGraphDebugDotPrint(graph, "/workspace/benchmarks/graph.dot", 0);

    return diffs;
}
