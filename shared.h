
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
