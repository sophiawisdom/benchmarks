#include <torch/extension.h>

#include <vector>

#include<tuple>
using namespace std;

// CUDA forward declarations

int bench_global(
    int *outs,
    int op_arg,
    int blocks,
    int threads,
    int dtype
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int benchmark_entrypoint(int dtype,
    torch::Tensor outs,
    int op,
    int shmem_size,
    int threads
) {
    throw std::logic_error("bench_global not yet implemented!");
    CHECK_INPUT(outs);

    assert(outs.dtype() == torch::kInt32);
    return bench_global((int *)outs.data_ptr(), op, shmem_size, threads, dtype);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bench_global", &benchmark_entrypoint, "cuda bench global. not implemented yet.");
}