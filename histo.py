from torch.utils.cpp_extension import load
import statistics
bench_histogram = load(name="bench_histogram", sources=["cpp/bench_histogram_entry.cpp", "cpp/bench_histogram.cu"], extra_cuda_cflags=["-arch=sm_89", "--keep", "--keep-dir", "/workspace/benchmark/temp"], verbose=True)
print(bench_histogram)

import torch
blocks = torch.cuda.get_device_properties(0).multi_processor_count
outs = torch.zeros((1024*32), dtype=torch.int32, device="cuda")

for bins in (1, 2, 4, 8, 16, 32, 64, 128, 256):
    for items_per_thread in (32, 128, 256, 1024, 65536,):
        for n_threads in (128,256):
            clocks = []
            for i in range(100):
                outs.zero_()
                result = bench_histogram.bench_histogram(outs, bins, items_per_thread, n_threads)
                if result == 1:
                    print(f"No kernel for ")
                    break

                outs_list = [int(a) for a in outs[:(n_threads//32)*blocks].tolist()]
                if outs_list.count(-1) != 0:
                    print(f"FOUND {outs_list.count(-1)} -1 VALUES!!!")
                    print(outs_list)
                    raise AssertionError
                elif outs_list.count(0) != 0:
                    print(f"FOUND {outs_list.count(0)} 0 VALUES")
                    print(outs_list)
                    raise AssertionError

                clocks.extend(outs_list)
            if not clocks: continue
            mean = int(statistics.mean(clocks))
            print(f"FOR {bins=}\t{items_per_thread=}\t{n_threads=}\tcycles/item: {int(mean/items_per_thread/32)}")
            # print(f"{outs_list[:100]}")
            # results[n_threads][f"{op_name}_{dtype}_{strat}"].append(mean)
            if mean < 1000:
                pass # print("\n\nTHIS OP AND DTYPE ARE PROBABLY EMPTY\n\n")