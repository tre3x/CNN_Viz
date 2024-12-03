import torch
import numpy as np
import torch.nn.functional as F


import torch
import numpy as np
import torch.nn.functional as F

def check(computed, input_tensor, kernel, bias, padding=0):
    # Convert input tensor to PyTorch tensor
    input_torch = torch.tensor(input_tensor.transpose(2, 0, 1)).unsqueeze(0)
    kernel_torch = torch.tensor(kernel.transpose(3, 2, 0, 1))
    bias_torch = torch.tensor(bias)

    # Determine padding for PyTorch
    if isinstance(padding, str):
        if padding.lower() == 'valid':
            padding_py = 0
        elif padding.lower() == 'same':
            padding_py = 'same'
        else:
            raise ValueError("Invalid padding string. Use 'valid' or 'same'.")
    elif isinstance(padding, int):
        padding_py = padding
    elif isinstance(padding, tuple) and len(padding) == 2:
        padding_py = padding
    else:
        raise ValueError("Padding must be 'valid', 'same', int, or tuple of two ints.")

    # Perform convolution using PyTorch with specified padding
    output_torch = F.conv2d(input_torch, kernel_torch, bias=bias_torch, padding=padding_py)

    # Convert PyTorch output back to numpy format
    output_torch_np = output_torch.squeeze(0).permute(1, 2, 0).numpy()

    # Compare CUDA and PyTorch results
    np.testing.assert_allclose(computed, output_torch_np, rtol=1e-5, atol=1e-5)
    print("CUDA and PyTorch outputs match!")
