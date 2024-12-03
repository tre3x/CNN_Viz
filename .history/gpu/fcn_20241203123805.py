# fully_connected_cuda.py

from numba import cuda
import numpy as np
from tests import fcn

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
    N_out = d_weights.shape[1]
    d_output = cuda.device_array(N_out, dtype=np.float32)

    # Calculate grid size
    blocks_per_grid = (N_out + threads_per_block - 1) // threads_per_block

    # Launch kernel
    fully_connected_kernel[blocks_per_grid, threads_per_block](d_input, d_weights, d_bias, d_output)

    return d_output  # Return device array

if __name__=="__main__":

    input_tensor = np.random.rand(100,).astype(np.float32)

    weights = np.random.rand(100, 200).astype(np.float32)

    bias = np.random.rand(200).astype(np.float32)
    
    d_input = cuda.to_device(input_tensor)
    d_weights = cuda.to_device(weights)
    d_bias = cuda.to_device(bias)

    output_cuda = fully_connected_layer(d_input, d_weights, d_bias)
    fcn.check(output_cuda, input_tensor, kernel, bias)
