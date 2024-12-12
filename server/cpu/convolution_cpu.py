import numpy as np
from scipy.signal import convolve2d

def convolution_layer(input_tensor, kernel, bias, padding='same'):
    """
    Performs convolution on the CPU.

    Parameters:
    input_tensor: Input tensor (H_in, W_in, C_in)
    kernel: Convolution kernels (kH, kW, C_in, C_out)
    bias: Bias for each output channel (C_out)
    padding: Padding mode ('valid', 'same')

    Returns:
    output_tensor: Output tensor (H_out, W_out, C_out)
    """
    H_in, W_in, C_in = input_tensor.shape
    kH, kW, kernel_C_in, C_out = kernel.shape

    if kernel_C_in != C_in:
        raise ValueError("Kernel depth does not match input channels")

    if padding.lower() == 'valid':
        pad_h = pad_w = 0
    elif padding.lower() == 'same':
        pad_h = (kH - 1) // 2
        pad_w = (kW - 1) // 2
    else:
        raise ValueError("Invalid padding string. Use 'valid' or 'same'.")

    padded_input = np.pad(input_tensor, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

    output_tensor = np.zeros((H_in, W_in, C_out), dtype=np.float32)

    for oc in range(C_out):
        for ic in range(C_in):
            output_tensor[:, :, oc] += convolve2d(padded_input[:, :, ic], kernel[:, :, ic, oc], mode='valid') 
        output_tensor[:, :, oc] += bias[oc]

    return output_tensor