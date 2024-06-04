from torch.utils.cpp_extension import load
import time
bench_l2 = load(name="bench_l2", sources=["l2.cpp", "l2_partitions.cu"], extra_cuda_cflags=["-arch=sm_89", "--keep", "--keep-dir", "/workspace/benchmarks/temp"], verbose=True)
print(bench_l2)

global_results, local_results = bench_l2.bench(0, 108, 128, 1)
#print(global_results)

summed_results = local_results.sum(axis=3).sum(axis=0) # sum b/w iterations and warps

print(f"{local_results[0, 0, :, 0]=}")

print(f"{summed_results[0]}")
print(f"{summed_results[:, 0]}")

'''
import matplotlib.pyplot as plt
import torch

summed_results = local_results.to(torch.int64).sum(axis=3).sum(axis=0) # sum b/w iterations and warps

for i in range(local_results.shape[0]):
    # Create the heatmap
    plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
    plt.imshow(local_results[i].to(torch.int64).sum(axis=2).numpy(), cmap='viridis')  # Choose a colormap that fits your preference
    plt.colorbar()  # Adds a colorbar to the side
    plt.title('between-SM L2 access latency')
    plt.show()
'''
