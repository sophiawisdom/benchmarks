#include <assert.h>

#include "curand_kernel.h"
#include "cuda_fp16.h"
#include "cuda_bf16.h"

#include <cmath>

#include "stdio.h"

#include <bit>

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



#define ATOMIC_OP_FLOAT(loc, val, operation)                                  \
    {                                                                         \
        unsigned int *int_loc = reinterpret_cast<unsigned int*>(loc);         \
        unsigned int old_int = __float_as_int(*loc);                          \
        unsigned int assumed_int;                                             \
        unsigned int new_int;                                                 \
        do {                                                                  \
            assumed_int = old_int;                                            \
            new_int = __float_as_int(operation(__int_as_float(assumed_int), val)); \
            old_int = atomicCAS(int_loc, assumed_int, new_int);               \
        } while (assumed_int != old_int);                                     \
        return __int_as_float(old_int);                                       \
    }

#define ATOMIC_OP_DOUBLE(loc, val, operation)                                 \
    {                                                                         \
        unsigned long long *long_loc = reinterpret_cast<unsigned long long*>(loc); \
        unsigned long long old_long = __double_as_longlong(*loc);             \
        unsigned long long assumed_long;                                      \
        unsigned long long new_long;                                          \
        do {                                                                  \
            assumed_long = old_long;                                          \
            new_long = __double_as_longlong(operation(__longlong_as_double(assumed_long), val)); \
            old_long = atomicCAS(long_loc, assumed_long, new_long);           \
        } while (assumed_long != old_long);                                   \
        return __longlong_as_double(old_long);                                \
    }

#define ATOMIC_OP_GENERAL(loc, val, operation)                                 \
if constexpr (std::is_same<dtype, float>::value) {\
    ATOMIC_OP_FLOAT(loc, val, operation)\
} else if constexpr (std::is_same<dtype, double>::value) {\
    ATOMIC_OP_DOUBLE(loc, val, operation)\
} else {                                                                      \
    dtype old = *loc;\
    dtype assumed;\
    do {\
        assumed = old;\
        old = atomicCAS(loc, assumed, operation(assumed, val));\
    } while (assumed != old);\
    return old;\
}

template<typename dtype>
__device__ __forceinline__ dtype sophiaAtomicMul(dtype *loc, dtype val) {
    ATOMIC_OP_GENERAL(loc, val, [] __device__ (dtype a, dtype b) { return a * b; });
}

template<typename dtype>
__device__ __forceinline__ dtype sophiaManualAtomicAdd(dtype *loc, dtype val) {
    ATOMIC_OP_GENERAL(loc, val, [] __device__ (dtype a, dtype b) { return a + b; });
}

template<typename dtype, int shmem_size>
__device__ __forceinline__ void warpcoalesced_add(dtype *out, dtype rand) {
    if constexpr(std::is_same<dtype, unsigned int>::value) {
        if constexpr (shmem_size == 1) {
            unsigned int result;
            asm("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(result) : "r"(rand));
            if ((threadIdx.x % 32) == 0) {
                atomicAdd(out, result);
            }
        } else if constexpr (shmem_size == 2) {
            // do threads that don't participate in the reduction get the result?
            unsigned int result;
            asm("redux.sync.add.u32 %0, %1, 0xaaaaaaaa;" : "=r"(result) : "r"(rand));
            unsigned int result2;
            asm("redux.sync.add.u32 %0, %1, 0x55555555;" : "=r"(result2) : "r"(rand));
            if ((threadIdx.x % 32) == 0) {
                atomicAdd(out, result);
            }
            if ((threadIdx.x % 32) == 1) {
                atomicAdd(out, result2);
            }
        } else {
            unsigned int sum = rand;
            if constexpr (shmem_size == 4) {
                sum += __shfl_xor_sync(0xffffffff, sum, 4);
                sum += __shfl_xor_sync(0xffffffff, sum, 8);
                sum += __shfl_xor_sync(0xffffffff, sum, 16);
            } else if constexpr (shmem_size == 8) {
                sum += __shfl_xor_sync(0xffffffff, sum, 8);
                sum += __shfl_xor_sync(0xffffffff, sum, 16);
            } else if constexpr (shmem_size == 16) {
                sum += __shfl_xor_sync(0xffffffff, sum, 16);
            }

            if ((threadIdx.x % 32) < shmem_size) {
                atomicAdd(out, sum);
            }
        }
    } else if constexpr (std::is_same<dtype, float>::value) {
        float sum = rand;
        if constexpr (shmem_size == 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, 1);
            sum += __shfl_xor_sync(0xffffffff, sum, 2);
            sum += __shfl_xor_sync(0xffffffff, sum, 4);
            sum += __shfl_xor_sync(0xffffffff, sum, 8);
            sum += __shfl_xor_sync(0xffffffff, sum, 16);
        } else if constexpr (shmem_size == 2) {
            sum += __shfl_xor_sync(0xffffffff, sum, 2);
            sum += __shfl_xor_sync(0xffffffff, sum, 4);
            sum += __shfl_xor_sync(0xffffffff, sum, 8);
            sum += __shfl_xor_sync(0xffffffff, sum, 16);
        } else if constexpr (shmem_size == 4) {
            sum += __shfl_xor_sync(0xffffffff, sum, 4);
            sum += __shfl_xor_sync(0xffffffff, sum, 8);
            sum += __shfl_xor_sync(0xffffffff, sum, 16);
        } else if constexpr (shmem_size == 8) {
            sum += __shfl_xor_sync(0xffffffff, sum, 8);
            sum += __shfl_xor_sync(0xffffffff, sum, 16);
        } else if constexpr (shmem_size == 16) {
            sum += __shfl_xor_sync(0xffffffff, sum, 16);
        }

        if ((threadIdx.x % 32) < shmem_size) {
            atomicAdd(out, sum);
        }
    }
}


template<typename dtype, Operation op, Strategy strat, int const_shmem_size>
__global__ void bench(int *outs, int shmem_size) {
    __shared__ dtype data[1024];

    unsigned int seed = threadIdx.x;
    curandState_t state;
    curand_init(seed, 0, 0, &state);

    for (int i = 0; i < shmem_size; i++) {
        if constexpr (std::is_same<dtype, __half2>::value) { // half2 needs different initializer
            data[i] = __floats2half2_rn(0, 0);
        } else if constexpr (op == ADD_NOCHANGE) {
            data[i] = INFINITY; // improves speed significantly
        } else {
            data[i] = 0;
        }
    }
    int our_index = threadIdx.x % shmem_size;
    if constexpr (strat == RAND) {
        our_index = curand(&state) % shmem_size;
    }

    __syncthreads();

    unsigned int start;
    asm volatile("mov.u32 %0, %clock;" : "=r"(start));

    for (int i = 0; i < 512; i++) {
        int rand_val = (op == INC) ? 1 : 15; // curand(&state);
        dtype rand = curand_cast<dtype>(rand_val);
        if constexpr (op == ADD || op == INC || op == ADD_NOCHANGE) {
            atomicAdd(&data[our_index], rand);
        } else if constexpr (op == MAX && (std::is_same<dtype, unsigned int>::value || std::is_same<dtype, unsigned long long>::value)) {
            atomicMax(&data[our_index], rand);
        } else if constexpr (op == MAX && (std::is_same<dtype, float>::value)) {
            atomicMaxFloat(&data[our_index], rand);
        } else if constexpr (op == XOR && (std::is_same<dtype, unsigned int>::value || std::is_same<dtype, unsigned long long>::value)) {
            atomicXor(&data[our_index], rand);
        } else if constexpr ((op == OR) && (std::is_same<dtype, unsigned int>::value || std::is_same<dtype, unsigned long long>::value)) {
            atomicOr(&data[our_index], rand);
        } else if constexpr (op == EXCH && (std::is_same<dtype, unsigned int>::value || std::is_same<dtype, unsigned long long>::value)) {
            atomicExch(&data[our_index], rand);
        } else if constexpr (op == MUL) {
            sophiaAtomicMul<dtype>(&data[our_index], rand);
        } else if constexpr (op == MANUAL_ADD) {
            sophiaManualAtomicAdd<dtype>(&data[our_index], rand);
        } else if constexpr (op == ADD_WARPCOALESCED) {
            static_assert(strat == TIDX);
            warpcoalesced_add<dtype, const_shmem_size>(&data[our_index], rand);
        }
    }

    unsigned int end;
    asm volatile("mov.u32 %0, %clock;" : "=r"(end));
    unsigned long long diff = end - start;

    if (threadIdx.x % 32 == 0) {
        outs[(blockIdx.x * blockDim.x + threadIdx.x)/32] = (int)diff;
    }
}

#define WARPCOALESCED(size) if (op == ADD_WARPCOALESCED && dtype == 0 && strat == 0 && shmem_size == size) {\
        kernel = &bench<float, ADD_WARPCOALESCED, TIDX, size>;\
    } else if (op == ADD_WARPCOALESCED && dtype == 2 && strat == 0) {\
        kernel = &bench<unsigned int, ADD_WARPCOALESCED, TIDX, size>;\
    }\


int bench_shared(
    int *outs,
    int op_arg,
    int shmem_size,
    int threads,
    int dtype,
    int strat
) {
    Operation op = (Operation)op_arg;
    using kernel_ptr = void(*)(int*, int);
    kernel_ptr kernel = nullptr;

    #define ASSIGN_KERNEL(DTYPE, TYPE_ID, OP) if (dtype == TYPE_ID && op == OP) {\
        if (strat == 0) {\
            kernel = &bench<DTYPE, OP, TIDX, 0>;\
        } else if (strat == 1) {\
            kernel = &bench<DTYPE, OP, RAND, 0>;\
        }\
    }

    ASSIGN_KERNEL(unsigned int, 1, ADD);
    ASSIGN_KERNEL(float, 0, ADD);
    ASSIGN_KERNEL(double, 3, ADD);
    ASSIGN_KERNEL(unsigned long long, 4, ADD);
    ASSIGN_KERNEL(__half2, 2, ADD);

    ASSIGN_KERNEL(unsigned int, 1, INC);

    ASSIGN_KERNEL(unsigned int, 1, MAX);
    ASSIGN_KERNEL(float, 0, MAX);
    ASSIGN_KERNEL(unsigned long long, 4, MAX);

    ASSIGN_KERNEL(unsigned int, 1, XOR);
    ASSIGN_KERNEL(unsigned long long, 4, XOR);

    ASSIGN_KERNEL(unsigned int, 1, OR);
    ASSIGN_KERNEL(unsigned long long, 4, OR);

    ASSIGN_KERNEL(unsigned int, 1, EXCH);
    ASSIGN_KERNEL(unsigned long long, 4, EXCH);

    ASSIGN_KERNEL(unsigned int, 1, MUL);
    ASSIGN_KERNEL(float, 0, MUL);
    ASSIGN_KERNEL(double, 3, MUL);
    ASSIGN_KERNEL(unsigned long long, 4, MUL);

    ASSIGN_KERNEL(unsigned int, 1, MANUAL_ADD);
    ASSIGN_KERNEL(float, 0, MANUAL_ADD);
    ASSIGN_KERNEL(double, 3, MANUAL_ADD);
    ASSIGN_KERNEL(unsigned long long, 4, MANUAL_ADD);

    ASSIGN_KERNEL(float, 0, ADD_NOCHANGE);
    ASSIGN_KERNEL(double, 3, ADD_NOCHANGE);

    WARPCOALESCED(1);
    WARPCOALESCED(2);
    WARPCOALESCED(4);
    WARPCOALESCED(8);
    WARPCOALESCED(16);
    WARPCOALESCED(32);
    WARPCOALESCED(64);
    WARPCOALESCED(128);
    if (op == ADD_WARPCOALESCED &&
        (shmem_size != 1) && (shmem_size != 2) && (shmem_size != 4) &&
        (shmem_size != 8) && (shmem_size != 16) && (shmem_size != 32) &&
        (shmem_size != 64) && (shmem_size != 128)) {
        printf("Couldn't run warpcoalesced op because shmem_size not in (1,2,4,8,16,32,64,128).\n");
        return 1;
    }

    if (kernel) {
        kernel<<<128, threads>>>(outs, shmem_size);
        return 0;
    } else {
        printf("RUNNING NOTHING for op=%d dtype=%d\n", op, dtype);
        return 1;
    }
}