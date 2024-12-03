# fully_connected_cuda.py

from numba import cuda
import numpy as np

@cuda.jit
def fully_connected_kernel(d_input, d_weights, d_bias, d_output):
    """
    CUDA kernel for fully connected layer (dense layer).

    Parameters:
    d_input: Flattened input tensor on the device (N_in,)
    d_weights: Weights matrix on the device (N_in, N_out)
    d_bias: Bias vector on the device (N_out,)
    d_output: Output vector on the device (N_out,)
    """
    idx = cuda.grid(1)
    N_in = d_weights.shape[0]
    N_out = d_weights.shape[1]

    if idx < N_out:
        sum = 0.0
        for i in range(N_in):
            sum += d_input[i] * d_weights[i, idx]
        d_output[idx] = sum + d_bias[idx]

def fully_connected_layer(d_input, d_weights, d_bias, threads_per_block=256):
    """
    Performs the fully connected layer operation on device arrays.

    Parameters:
    d_input: Flattened input tensor on the device (N_in,)
    d_weights: Weights matrix on the device (N_in, N_out)
    d_bias: Bias vector on the device (N_out,)
    threads_per_block: Number of threads per block for CUDA kernel

    Returns:
    d_output: Output vector on the device (N_out,)
    """
    N_out = d_weights.shape[1]
    d_output = cuda.device_array(N_out, dtype=np.float32)

    # Calculate grid size
    blocks_per_grid = (N_out + threads_per_block - 1) // threads_per_block

    # Launch kernel
    fully_connected_kernel[blocks_per_grid, threads_per_block](d_input, d_weights, d_bias, d_output)

    return d_output  # Return device array


