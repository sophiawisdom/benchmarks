#include <torch/extension.h>

#include <vector>

#include<tuple>
using namespace std;

int bench_histogram(
    int * outs,
    int bins,
    int items_per_thread,
    int threads
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int benchmark_entrypoint(torch::Tensor outs,
    int bins,
    int items_per_thread,
    int threads
) {
    CHECK_INPUT(outs);

    assert(outs.dtype() == torch::kInt32);
    return bench_histogram((int *)outs.data_ptr(), bins, items_per_thread, threads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bench_histogram", &benchmark_entrypoint, "cuda bench histogram");
}