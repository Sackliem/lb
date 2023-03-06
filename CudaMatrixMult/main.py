from numba import cuda, float32
import numba
import numpy
import math
import time

TPB = 16


@cuda.jit()
def matmul_gpu(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp


A = numpy.full((TPB * 500, TPB * 500), 3, numpy.float32)
B = numpy.full((TPB * 500, TPB * 500), 4, numpy.float32)

# start in GPU
A_global_mem = cuda.to_device(A)
B_global_mem = cuda.to_device(B)

C_global_mem = cuda.device_array((A.shape[0], B.shape[1]))

threads_per_block = (TPB, TPB)
blocks_per_grid_x = int(math.ceil(A.shape[0] / threads_per_block[0]))
blocks_per_grid_y = int(math.ceil(B.shape[1] / threads_per_block[1]))
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

print("start processing in GPU")
start_gpu = time.time()
matmul_gpu[blocks_per_grid, threads_per_block](A_global_mem, B_global_mem, C_global_mem)
cuda.synchronize()
end_gpu = time.time()
time_gpu = end_gpu - start_gpu
print("GPU time(Global memory):" + str(time_gpu))
C_global_gpu = C_global_mem.copy_to_host()
print(A, "\n", B, "\n", C_global_gpu)
