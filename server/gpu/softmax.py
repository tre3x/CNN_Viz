from numba import cuda, float32
import numpy as np
import math
from tests import softmax_test

@cuda.jit
def softmax_kernel(logits, max_logit, exp_logits, sum_exp_logits):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    # First pass: Compute exponentials and sum
    for i in range(idx, logits.size, stride):
        # Use math.exp instead of cuda.exp
        exp_val = math.exp(logits[i] - max_logit[0])
        exp_logits[i] = exp_val
        cuda.atomic.add(sum_exp_logits, 0, exp_val)
        
def softmax_activation(d_logits, threads_per_block=256):
    """
    Performs softmax activation on the device array and returns the output as a device array.

    Parameters:
    d_logits: Input logits tensor on the device (N_out,)
    threads_per_block: Number of threads per block for CUDA kernel

    Returns:
    d_output: Output probabilities tensor on the device (N_out,)
    """
    N = d_logits.size

    # Allocate device arrays
    d_exp_logits = cuda.device_array_like(d_logits)
    d_output = cuda.device_array_like(d_logits)
    max_logits = cuda.device_array(1, dtype=np.float32)
    sum_exp_logits = cuda.device_array(1, dtype=np.float32)
    sum_exp_logits[0] = 0.0

    # Compute max logit for numerical stability
    max_logit_host = np.array([d_logits.copy_to_host().max()], dtype=np.float32)
    cuda.to_device(max_logit_host, to=max_logits)

    # Calculate grid size
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    # Launch kernel to compute exponentials and sum
    softmax_kernel[blocks_per_grid, threads_per_block](d_logits, max_logits, d_exp_logits, sum_exp_logits)

    # Copy sum_exp_logits to host
    sum_exp_logits_host = sum_exp_logits.copy_to_host()
    sum_exp = sum_exp_logits[0]

    # Normalize exponentials
    d_output = d_exp_logits / sum_exp

    return d_output


if __name__=="__main__":
    input_tensor = np.random.rand(1000, ).astype(np.float32)
    d_input = cuda.to_device(input_tensor)
    
    output_cuda = softmax_activation(d_input)
    softmax_test.check(output_cuda, input_tensor)