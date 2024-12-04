from numba import cuda, float32
import numpy as np
import time
from tests import convolution_test

@cuda.jit
def conv_layer_kernel(d_input, d_kernel, d_bias, d_output):
    """
    CUDA kernel for convolution layer in CNN with padding.
    
    Parameters:
    d_input: Padded input tensor on the device (H_in + 2 * pad_h, W_in + 2 * pad_w, C_in)
    d_kernel: Convolution kernels on the device (kH, kW, C_in, C_out)
    d_bias: Bias for each output channel on the device (C_out)
    d_output: Output tensor on the device (H_out, W_out, C_out)
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
    kH, kW, C_in, C_out = d_kernel.shape

    # Output dimensions
    H_out, W_out, C_out = d_output.shape

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
                            d_input[in_row, in_col, ic]
                            * d_kernel[kh, kw, ic, oc]
                        )
            # Add bias and write to output
            d_output[row, col, oc] = result + d_bias[oc]

def convolution_layer(d_input, d_kernel, d_bias, padding='same', threads_per_block=(16, 16)):
    """
    Performs convolution on the device arrays and returns the output as a device array.
    
    Parameters:
    d_input: Input tensor on the device (H_in, W_in, C_in)
    d_kernel: Convolution kernels on the device (kH, kW, C_in, C_out)
    d_bias: Bias for each output channel on the device (C_out)
    padding: Padding mode ('valid', 'same', int, or tuple)
    threads_per_block: Threads per block for CUDA kernel
    
    Returns:
    d_output: Output tensor on the device (H_out, W_out, C_out)
    """
    # Get input dimensions
    H_in, W_in, C_in = d_input.shape
    kH, kW, kernel_C_in, C_out = d_kernel.shape

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
            pad_h = (kH - 1) // 2
            pad_w = (kW - 1) // 2
        else:
            raise ValueError("Invalid padding string. Use 'valid' or 'same'.")
    elif isinstance(padding, int):
        pad_h = pad_w = padding
    elif isinstance(padding, tuple) and len(padding) == 2:
        pad_h, pad_w = padding
    else:
        raise ValueError("Padding must be 'valid', 'same', int, or tuple of two ints.")

    # Pad the input tensor on the device
    if pad_h > 0 or pad_w > 0:
        H_padded = H_in + 2 * pad_h
        W_padded = W_in + 2 * pad_w
        C_in = d_input.shape[2]

        d_input_padded = cuda.device_array((H_padded, W_padded, C_in), dtype=np.float32)
        d_input_padded[pad_h:pad_h+H_in, pad_w:pad_w+W_in, :] = d_input

        # Zero-padding for the borders
        # (Borders are already zero-initialized in device_array)
    else:
        d_input_padded = d_input

    # Calculate output dimensions
    H_out = (H_in + 2 * pad_h - kH) // 1 + 1  # Assuming stride=1
    W_out = (W_in + 2 * pad_w - kW) // 1 + 1

    # Allocate device memory for output
    d_output = cuda.device_array((H_out, W_out, C_out), dtype=np.float32)

    # Calculate grid size
    blocks_per_grid_x = (W_out + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (H_out + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch kernel
    conv_layer_kernel[blocks_per_grid, threads_per_block](d_input_padded, d_kernel, d_bias, d_output)

    return d_output  # Return device array



if __name__=="__main__":

    input_tensor = np.random.rand(224, 224, 3).astype(np.float32)

    kernel = np.random.rand(3, 3, 3, 64).astype(np.float32)

    bias = np.random.rand(64).astype(np.float32)
    
    padding = "same"
    
    d_input = cuda.to_device(input_tensor)
    d_kernel = cuda.to_device(kernel)
    d_bias = cuda.to_device(bias)

    output_cuda = convolution_layer(d_input, d_kernel, d_bias, padding)
    convolution_test.check(output_cuda, input_tensor, kernel, bias, padding)
    
    


