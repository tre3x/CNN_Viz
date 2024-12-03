from numba import cuda, float32
import numpy as np
import time
from tests import pooling

@cuda.jit
def pooling_kernel(d_input, d_output, pool_size, stride, mode_flag):
    """
    CUDA kernel for pooling layer.
    
    Parameters:
    d_input: Input tensor on the device (H_in, W_in, C)
    d_output: Output tensor on the device (H_out, W_out, C)
    pool_size: Size of the pooling window
    stride: Stride of the pooling operation
    mode_flag: 0 for max pooling, 1 for average pooling
    """
    # Get thread indices
    i, j, c = cuda.grid(3)

    H_out, W_out, C = d_output.shape

    if i < H_out and j < W_out and c < C:
        H_in, W_in, _ = d_input.shape
        start_i = i * stride
        start_j = j * stride
        end_i = min(start_i + pool_size, H_in)
        end_j = min(start_j + pool_size, W_in)

        if mode_flag == 0:  # Max pooling
            max_val = float('-inf')
            for m in range(start_i, end_i):
                for n in range(start_j, end_j):
                    val = d_input[m, n, c]
                    if val > max_val:
                        max_val = val
            d_output[i, j, c] = max_val
        else:  # Average pooling
            sum_val = 0.0
            count = 0
            for m in range(start_i, end_i):
                for n in range(start_j, end_j):
                    sum_val += d_input[m, n, c]
                    count += 1
            d_output[i, j, c] = sum_val / count

def pooling_layer(d_input, pool_size=2, stride=2, mode='max', threads_per_block=(8, 8, 8)):
    """
    Performs pooling on the device array and returns the output as a device array.
    
    Parameters:
    d_input: Input tensor on the device (H_in, W_in, C)
    pool_size: Size of the pooling window
    stride: Stride of the pooling operation
    mode: 'max' or 'avg' for max pooling or average pooling
    threads_per_block: Threads per block for CUDA kernel
    
    Returns:
    d_output: Output tensor on the device (H_out, W_out, C)
    """
    H_in, W_in, C = d_input.shape

    # Calculate output dimensions
    H_out = (H_in - pool_size) // stride + 1
    W_out = (W_in - pool_size) // stride + 1

    # Allocate device memory for output
    d_output = cuda.device_array((H_out, W_out, C), dtype=np.float32)

    # Pooling mode
    mode_flag = 0 if mode == 'max' else 1

    # Calculate grid size
    blocks_per_grid_x = (H_out + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (W_out + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid_z = (C + threads_per_block[2] - 1) // threads_per_block[2]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

    # Launch kernel
    pooling_kernel[blocks_per_grid, threads_per_block](d_input, d_output, pool_size, stride, mode_flag)

    return d_output  # Return device array

if __name__=="__main__":

    # Input tensor: 32x32x3 (e.g., feature map from convolution layer)
    input_tensor = np.random.rand(224, 224, 128).astype(np.float32)

    # Pooling parameters
    pool_size = 4
    stride = 4


    d_input = cuda.to_device(input_tensor)
    
    # Perform max pooling
    output_max_pool = pooling_layer(d_input, pool_size, stride, mode='max')
    pooling.check(output_max_pool, input_tensor, pool_size, stride, mode="max")

    output_avg_pool = pooling_layer(d_input, pool_size, stride, mode='avg')
    pooling.check(output_avg_pool, input_tensor, pool_size, stride, mode="avg")
    
    