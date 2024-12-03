import torch
import numpy as np
import torch.nn.functional as F

def check(computed, input_tensor, weights, bias):
    """
    Test script to validate the output of the fully connected layer.

    Parameters:
    computed: Output from your CUDA fully connected layer (numpy array)
    input_tensor: Input numpy array to the fully connected layer
    weights: Weights matrix used in the fully connected layer
    bias: Bias vector used in the fully connected layer
    """
    # Convert input tensor to PyTorch tensor
    input_torch = torch.tensor(input_tensor, dtype=torch.float32)

    # Convert weights and bias to PyTorch tensors
    # Note: In PyTorch, the weight matrix shape is (out_features, in_features)
    weights_torch = torch.tensor(weights.T, dtype=torch.float32)  # Transpose to match PyTorch format
    bias_torch = torch.tensor(bias, dtype=torch.float32)

    # Perform fully connected operation using PyTorch
    output_torch = F.linear(input_torch, weights_torch, bias_torch)

    # Convert PyTorch output to numpy array
    output_torch_np = output_torch.detach().numpy()

    # Compare CUDA and PyTorch results
    np.testing.assert_allclose(computed, output_torch_np, rtol=1e-5, atol=1e-5)
    print("Fully Connected Layer: CUDA and PyTorch outputs match!")
