#include <torch/extension.h>

#include <vector>

#include <cuda_runtime.h>

#include<tuple>
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

// CUDA forward declarations

float *bench_l2(
    int *outs,
    int op_arg,
    int blocks,
    int threads,
    int iterations
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

size_t roundUpToPageSize(int size) {
    size_t remainder = size % 16384;
    if (remainder == 0) {
        return size; // Already a multiple of page size
    }
    return size + 16384 - remainder; // Round up to the next multiple
}

std::tuple<torch::Tensor, torch::Tensor> benchmark_entrypoint(
    int op_arg,
    int blocks,
    int threads,
    int iterations
) {
    // assert(blocks == 128);
    int out_size = blocks * blocks * 4; // per iteration

    assert(threads == 128); // change the 4s

    int *cpu_outs;
    CUDA_CHECK(cudaMallocHost(&cpu_outs, roundUpToPageSize(iterations * out_size * sizeof(int))));

    auto options = torch::TensorOptions().dtype(torch::kInt32);
    // assert(blocks == 128);
    torch::Tensor out_tensor = torch::from_blob(
        cpu_outs,
        {iterations, blocks, blocks, 4}, {blocks*blocks*4, blocks*4, 4, 1}, // shape/stride
        [](void* ptr) { // deleter
            cudaFreeHost(ptr);
        },
        options
    );

    assert(out_tensor.dtype() == torch::kInt32);
    float *global_results = bench_l2(cpu_outs, op_arg, blocks, threads, iterations);
    if (global_results == nullptr) {
        throw std::exception();
    }

    auto global_options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor global_tensor = torch::from_blob(
        global_results,
        {iterations}, {1}, // shape/stride
        [](void* ptr) { // deleter
            free(ptr);
        },
        global_options
    );

    CUDA_CHECK(cudaDeviceSynchronize());
    return make_tuple(global_tensor, out_tensor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bench", &benchmark_entrypoint, "cuda bench");
}