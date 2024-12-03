from numba import cuda, float32
import numpy as np
import time
from tests import pooling

@cuda.jit
def pool_layer(input_tensor, output_tensor, pool_size, stride, mode):
    """
    CUDA kernel for pooling layer without shared memory.
    """
    # Thread and block indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    # Global thread position
    row = by * cuda.blockDim.y + ty
    col = bx * cuda.blockDim.x + tx

    # Input and output dimensions
    H, W, C = input_tensor.shape
    H_out, W_out, C = output_tensor.shape

    # Loop over channels
    for c in range(C):
        if row < H_out and col < W_out:
            start_row = row * stride
            start_col = col * stride
            end_row = min(start_row + pool_size, H)
            end_col = min(start_col + pool_size, W)

            if mode == 0:  # Max pooling
                max_val = float('-inf')
                for i in range(start_row, end_row):
                    for j in range(start_col, end_col):
                        max_val = max(max_val, input_tensor[i, j, c])
                output_tensor[row, col, c] = max_val
            elif mode == 1:  # Average pooling
                sum_val = 0.0
                count = 0
                for i in range(start_row, end_row):
                    for j in range(start_col, end_col):
                        sum_val += input_tensor[i, j, c]
                        count += 1
                output_tensor[row, col, c] = sum_val / count


def pooling_layer(input_tensor, pool_size, stride, mode='max', threads_per_block=(16, 16)):
    """
    Host function for pooling layer.
    
    Parameters:
    input_tensor: Input numpy array (H x W x C)
    pool_size: Pool size (int)
    stride: Stride (int)
    mode: 'max' or 'avg' for max pooling or average pooling
    threads_per_block: Threads per block for CUDA kernel
    
    Returns:
    output_tensor: Output numpy array (H_out x W_out x C)
    """
    H, W, C = input_tensor.shape

    # Compute output dimensions
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    # Output tensor
    output_tensor = np.zeros((H_out, W_out, C), dtype=np.float32)

    # Allocate device memory
    d_input = cuda.to_device(input_tensor)
    d_output = cuda.device_array_like(output_tensor)

    # Pooling mode
    mode_flag = 0 if mode == 'max' else 1

    # Calculate grid size
    blocks_per_grid_x = (W_out + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (H_out + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch kernel
    pool_layer[blocks_per_grid, threads_per_block](d_input, d_output, pool_size, stride, mode_flag)

    # Copy result back to host
    output_tensor = d_output.copy_to_host()
    return output_tensor

if __name__=="__main__":

    # Input tensor: 32x32x3 (e.g., feature map from convolution layer)
    input_tensor = np.random.rand(224, 224, 128).astype(np.float32)

    # Pooling parameters
    pool_size = 4
    stride = 4

    # Perform max pooling
    output_max_pool = pooling_layer(input_tensor, pool_size, stride, mode='max')
    pooling.check(output_max_pool, input_tensor, pool_size, stride, mode="max")

    output_avg_pool = pooling_layer(input_tensor, pool_size, stride, mode='avg')
    pooling.check(output_avg_pool, input_tensor, pool_size, stride, mode="avg")
    
    