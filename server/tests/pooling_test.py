import torch
import numpy as np
import torch.nn.functional as F


def check(computed, input_tensor, pool_size, stride, mode):
    # Convert input tensor to PyTorch tensor
    input_torch = torch.tensor(input_tensor.transpose(2, 0, 1)).unsqueeze(0)

    # Select pooling mode
    if mode == 'max':
        pool_layer = torch.nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=0)
    elif mode == 'avg':
        pool_layer = torch.nn.AvgPool2d(kernel_size=pool_size, stride=stride, padding=0)
    else:
        raise ValueError("Invalid mode. Use 'max' or 'avg'.")

    # Perform pooling using PyTorch
    output_torch = pool_layer(input_torch)

    # Convert PyTorch output back to numpy format
    output_torch_np = output_torch.squeeze(0).permute(1, 2, 0).numpy()

    # Compare CUDA and PyTorch results
    np.testing.assert_allclose(computed, output_torch_np, rtol=1e-5, atol=1e-5)
    print(f"{mode.capitalize()} Pooling: CUDA and PyTorch outputs match!")