#include <torch/extension.h>

#include <vector>

#include<tuple>
using namespace std;

// CUDA forward declarations

int bench_shared(
    int * outs,
    int op,
    int shmem_size,
    int threads,
    int dtype,
    int strat
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int benchmark_entrypoint(int dtype,
    torch::Tensor outs,
    int op,
    int shmem_size,
    int threads,
    int strat
) {
    CHECK_INPUT(outs);

    assert(outs.dtype() == torch::kInt32);
    return bench_shared((int *)outs.data_ptr(), op, shmem_size, threads, dtype, strat);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bench_shared", &benchmark_entrypoint, "cuda bench");
}