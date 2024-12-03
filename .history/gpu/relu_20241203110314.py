from numba import cuda
import numpy as np
import time
from tests import relu

@cuda.jit
def relu_kernel(input_tensor, output_tensor):
    """
    CUDA kernel for ReLU activation function.
    
    Parameters:
    input_tensor: Input feature map (H x W x C)
    output_tensor: Output feature map (H x W x C)
    """
    # Get thread indices
    i, j, k = cuda.grid(3)

    H, W, C = input_tensor.shape

    if i < H and j < W and k < C:
        val = input_tensor[i, j, k]
        output_tensor[i, j, k] = max(0.0, val)

def relu_activation(input_tensor, threads_per_block=(8,8,8)):
    """
    Host function for ReLU activation.
    
    Parameters:
    input_tensor: Input numpy array (H x W x C)
    threads_per_block: Threads per block for CUDA kernel
    
    Returns:
    output_tensor: Output numpy array (H x W x C)
    """
    H, W, C = input_tensor.shape

    # Output tensor
    output_tensor = np.empty_like(input_tensor)

    # Allocate device memory
    d_input = cuda.to_device(input_tensor)
    d_output = cuda.device_array_like(output_tensor)

    # Calculate grid size
    blocks_per_grid_x = (H + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (W + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid_z = (C + threads_per_block[2] - 1) // threads_per_block[2]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

    # Launch kernel
    relu_kernel[blocks_per_grid, threads_per_block](d_input, d_output)

    # Copy result back to host
    output_tensor = d_output.copy_to_host()
    return output_tensor

if __name__=="__main__":

    # Input tensor: 32x32x3 with negative and positive values
    input_tensor = np.random.randn(224, 224, 64).astype(np.float32)

    # Perform ReLU activation
    output_relu = relu_activation(input_tensor)

    # Validate the result
    relu.check(output_relu, input_tensor)
