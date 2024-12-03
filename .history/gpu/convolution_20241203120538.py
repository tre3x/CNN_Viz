from numba import cuda, float32
import numpy as np
import time
from tests import convolution

@cuda.jit
def conv_layer(input_tensor, kernel, bias, output_tensor):
    """
    CUDA kernel for convolution layer in CNN with padding.
    """
    # Thread and block indices
    tx = cuda.threadIdx.x  # Thread ID in block (x-axis)
    ty = cuda.threadIdx.y  # Thread ID in block (y-axis)
    bx = cuda.blockIdx.x   # Block ID (x-axis)
    by = cuda.blockIdx.y   # Block ID (y-axis)

    # Global thread position (output coordinates)
    row = by * cuda.blockDim.y + ty
    col = bx * cuda.blockDim.x + tx

    # Kernel dimensions
    kH, kW, C_in, C_out = kernel.shape

    # Output dimensions
    H_out, W_out, C_out = output_tensor.shape

    # Check if within output bounds
    if row < H_out and col < W_out:
        # Loop over output channels
        for oc in range(C_out):
            result = 0.0
            # Loop over input channels and kernel elements
            for ic in range(C_in):
                for kh in range(kH):
                    for kw in range(kW):
                        in_row = row + kh
                        in_col = col + kw
                        result += (
                            input_tensor[in_row, in_col, ic]
                            * kernel[kh, kw, ic, oc]
                        )
            # Add bias and write to output
            output_tensor[row, col, oc] = result + bias[oc]

def convolution_layer(d_input, d_kernel, d_bias, padding=0, threads_per_block=(16, 16)):
    """
    Host function for CNN convolution layer with flexible padding.
    """
    H_in, W_in, C_in = input_tensor.shape
    kH, kW, kernel_C_in, C_out = kernel.shape

    # Validate kernel dimensions
    if kernel_C_in != C_in:
        raise ValueError(
            f"Kernel depth ({kernel_C_in}) does not match input channels ({C_in})"
        )

    # Determine padding amounts
    if isinstance(padding, str):
        if padding.lower() == 'valid':
            pad_h = pad_w = 0
        elif padding.lower() == 'same':
            pad_h = ((H_in - 1) * 1 + kH - H_in) // 2  # Assuming stride=1
            pad_w = ((W_in - 1) * 1 + kW - W_in) // 2
        else:
            raise ValueError("Invalid padding string. Use 'valid' or 'same'.")
    elif isinstance(padding, int):
        pad_h = pad_w = padding
    elif isinstance(padding, tuple) and len(padding) == 2:
        pad_h, pad_w = padding
    else:
        raise ValueError("Padding must be 'valid', 'same', int, or tuple of two ints.")

    # Pad the input tensor
    if pad_h > 0 or pad_w > 0:
        input_padded = np.pad(
            input_tensor,
            ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode='constant',
            constant_values=0
        )
    else:
        input_padded = input_tensor

    # Calculate output dimensions
    H_out = (H_in + 2 * pad_h - kH) // 1 + 1  # Assuming stride=1
    W_out = (W_in + 2 * pad_w - kW) // 1 + 1

    output_tensor = np.zeros((H_out, W_out, C_out), dtype=np.float32)

    # Allocate device memory
    d_input = cuda.to_device(input_padded)
    d_kernel = cuda.to_device(kernel)
    d_bias = cuda.to_device(bias)
    d_output = cuda.device_array_like(output_tensor)

    # Calculate grid size
    blocks_per_grid_x = (W_out + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (H_out + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch kernel
    conv_layer[blocks_per_grid, threads_per_block](d_input, d_kernel, d_bias, d_output)

    # Copy result back to host
    output_tensor = d_output.copy_to_host()
    return output_tensor



if __name__=="__main__":

    input_tensor = np.random.rand(32, 32, 3).astype(np.float32)

    kernel = np.random.rand(3, 3, 3, 16).astype(np.float32)

    bias = np.random.rand(16).astype(np.float32)
    
    padding = "valid"

    output_cuda = convolution_layer(input_tensor, kernel, bias, padding)
    
    convolution.check(output_cuda, input_tensor, kernel, bias, padding)
    
    


