from numba import cuda
import numpy as np
import time
from tests import relu

@cuda.jit
def relu_kernel(d_input, d_output):
    """
    CUDA kernel for ReLU activation function.
    
    Parameters:
    d_input: Input tensor on the device
    d_output: Output tensor on the device
    """
    idx = cuda.grid(1)
    size = d_input.size

    if idx < size:
        val = d_input.flat[idx]
        d_output.flat[idx] = max(0.0, val)

def relu_activation(d_input, threads_per_block=256):
    """
    Performs ReLU activation on the device array and returns the output as a device array.
    
    Parameters:
    d_input: Input tensor on the device
    threads_per_block: Threads per block for CUDA kernel
    
    Returns:
    d_output: Output tensor on the device
    """
    # Allocate device memory for output
    d_output = cuda.device_array_like(d_input)

    # Flatten the input for 1D kernel
    size = d_input.size

    # Calculate grid size
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block

    # Launch kernel
    relu_kernel[blocks_per_grid, threads_per_block](d_input, d_output)

    return d_output  # Return device array

if __name__=="__main__":

    # Input tensor: 32x32x3 with negative and positive values
    input_tensor = np.random.randn(224, 224, 64).astype(np.float32)
    
    d_input = cuda.to_device(input_tensor)

    # Perform ReLU activation
    output_relu = relu_activation(input_tensor)

    # Validate the result
    relu.check(output_relu, input_tensor)
