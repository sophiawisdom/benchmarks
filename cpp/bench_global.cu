#include <stdio.h>

#include "cuda_fp16.h"
#include "cuda_bf16.h"

// shared b/w bench_global and bench_shared
enum Operation {
    ADD,
    INC,
    MAX,
    XOR,
    OR,
    EXCH,
    MUL,
    MANUAL_ADD,
    ADD_NOCHANGE,
    ADD_WARPCOALESCED
};

enum Strategy {
    TIDX,
    RAND
};

__forceinline__ __device__ __host__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x+b.x, a.y+b.y, a.z*b.z, a.w*b.w);
}

__device__ __forceinline__ float atomicMaxFloat(float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

template<typename dtype>
__device__ __forceinline__ dtype curand_cast(int val) {
    if constexpr (std::is_same<dtype, __half2>::value) {
        return __floats2half2_rn((float)val, (float)val);
    } else {
        return (dtype)val;
    }
}


template<typename dtype, Operation op>
__global__ void bench(void *scratchpad, int *outs) {
    // TODO: CONSIDER SMID split-l2
    return;
}


int bench_global(
    int *outs,
    int op_arg,
    int blocks,
    int threads,
    int dtype
) {
    Operation op = (Operation)op_arg;
    using kernel_ptr = void(*)(void*, int*);
    kernel_ptr kernel = nullptr;

    void *scratchpad;
    cudaMalloc(&scratchpad, 16384);
    cudaMemset(scratchpad, 0, 16384);

    #define ASSIGN_KERNEL(DTYPE, TYPE_ID, OP) if (dtype == TYPE_ID && op == OP) {kernel = &bench<DTYPE, OP>;}

    if (kernel) {
        kernel<<<blocks, 128>>>(scratchpad, outs);
        return 0;
    } else {
        printf("RUNNING NOTHING for op=%d dtype=%d\n", op, dtype);
        return 1;
    }

    cudaFree(scratchpad);
}