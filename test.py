from torch.utils.cpp_extension import load
import torch
import random
import statistics
import matplotlib.pyplot as plt

from collections import defaultdict

bench = load(name="bench", sources=["file.cpp", "bench_shared.cu", "bench_global.cu"], extra_cuda_cflags=["--keep", "--keep-dir", "/workspace/benchmark/temp", "--extended-lambda"], verbose=True)

outs = torch.zeros((1024*32), dtype=torch.int32, device="cuda")

results = defaultdict(lambda:defaultdict(list))

dtypes = ["float32", "uint32"]#, "half2", "double", "int64"]
ops = ["add", "inc", "max", "xor", "or", "exch", "mul", "manualadd", "add_nochange", "add_warpcoalesced"]
strats = ["TIDX"]#, "CURAND"]

for strat, strat_name in enumerate(strats):
    for op, op_name in enumerate(ops):
        if op_name not in ("add", "add_nochange", "add_warpcoalesced"): continue
        for dtype_enum, dtype in enumerate(dtypes):
            if bench.bench(dtype_enum, outs, op, 128, 128, 0) == 1:
                continue
            # if dtype not in ("int32", "inc", "float32"): continue
            for n_threads in (32, 64, 128, 256, 512, 1024):
                if n_threads not in (256,): continue
                for shmem_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                    # if shmem_size not in (1, 256): continue
                    if shmem_size > n_threads: continue
                    clocks = []
                    for i in range(100):
                        outs.zero_()
                        result = bench.bench(dtype_enum, outs, op, shmem_size, n_threads, strat)
                        if result == 1:
                            print(f"No kernel for ")
                            break

                        outs_list = [int(a) for a in outs[:n_threads*128].tolist()]
                        if outs_list.count(-1) != 0:
                            print(f"FOUND {outs_list.count(-1)} -1 VALUES!!!")
                            print(outs_list)
                            raise AssertionError
                        elif outs_list.count(0) != 0:
                            print(f"FOUND {outs_list.count(0)} 0 VALUES")
                            print(outs_list)
                            raise AssertionError

                        clocks.extend([int(a) for a in outs_list[::32]])
                    if not clocks: continue
                    mean = int(statistics.mean(clocks))
                    print(f"FOR {dtype=}\top={op_name}\tstrat={strat_name}\t{shmem_size=}\t{n_threads=}\tmean: {mean}")
                    results[n_threads][f"{op_name}_{dtype}"].append(mean)
                    if mean < 1000:
                        pass # print("\n\nTHIS OP AND DTYPE ARE PROBABLY EMPTY\n\n")

                # print(f"{results=}")

results = dict(results)
