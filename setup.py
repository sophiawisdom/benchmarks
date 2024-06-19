from setuptools import setup, Extension

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
)

args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "--extended-lambda",
        "-gencode", "arch=compute_70,code=sm_70",
        "-gencode", "arch=compute_80,code=sm_80",
    ],
}

extensions = [
    CUDAExtension("bench_global",["cpp/bench_global.cu", "cpp/bench_global_entry.cpp"], extra_compile_args=args),
    CUDAExtension("bench_shared", ["cpp/bench_shared.cu", "cpp/bench_shared_entry.cpp"], extra_compile_args=args),
    CUDAExtension("bench_l2",["cpp/bench_l2.cu", "cpp/bench_l2_entry.cpp"], extra_compile_args=args),
]

setup(
    name="cuda_benchmarks",
    version="1.0.0",
    packages=[],
    author="Sophia Wisdom",
    author_email="sophia.wisdom1999+remove_this_if_your_message_isnt_spam@gmail.com",
    description="Sophia's CUDA benchmarks",
    url="https://github.com/sophiawisdom/benchmarks",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=extensions,
    python_requires=">=3.10",
    install_requires=['pybind11>=2.5.0'],
    zip_safe=False,
    cmdclass={
        'build_ext': BuildExtension
    }
)