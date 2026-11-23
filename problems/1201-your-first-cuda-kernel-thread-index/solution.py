#include <cuda_runtime.h>
#include <vector>

__global__ void index_kernel(int* out, int n) {

    // Calculate global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Only write if index is within range
    if (i < n) {
        out[i] = i;
    }
}

std::vector<int> global_thread_indices(int n) {

    // Handle invalid size
    if (n <= 0) return {};

    // Allocate memory on GPU
    int* d_out;
    cudaMalloc(&d_out, n * sizeof(int));

    int threads = 256;

    // Calculate required number of blocks
    int blocks = (n + threads - 1) / threads;

    // Execute kernel on GPU
    index_kernel<<<blocks, threads>>>(d_out, n);

    // Create output vector on CPU
    std::vector<int> result(n);

    // Copy GPU result to CPU
    cudaMemcpy(
        result.data(),
        d_out,
        n * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    // Free GPU memory
    cudaFree(d_out);

    return result;
}
