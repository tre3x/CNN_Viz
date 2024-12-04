import torch
import numpy as np
import torch.nn.functional as F

def check(computed, input_tensor):
    # Convert input tensor to PyTorch tensor
    input_torch = torch.tensor(input_tensor, dtype=torch.float32)

    # Perform ReLU using PyTorch
    output_torch = F.relu(input_torch)

    # Convert PyTorch output back to numpy format
    output_torch_np = output_torch.numpy()

    # Compare CUDA and PyTorch results
    np.testing.assert_allclose(computed, output_torch_np, rtol=1e-5, atol=1e-5)
    print("ReLU activation: CUDA and PyTorch outputs match!")
