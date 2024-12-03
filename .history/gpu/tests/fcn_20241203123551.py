# tests/fully_connected_test.py

import numpy as np
import torch
import torch.nn.functional as F
from numba import cuda
from layers import FullyConnectedLayer

def test_fully_connected_layer():
    # Input size and output size
    in_features = 1024  # Example input size
    out_features = 256  # Example output size

    # Create test input
    input_tensor = np.random.randn(in_features).astype(np.float32)
    d_input = cuda.to_device(input_tensor)

    # Initialize FullyConnectedLayer
    fc_layer = FullyConnectedLayer(in_features, out_features)

    # Use the same weights and biases in PyTorch for comparison
    weight = torch.tensor(fc_layer.weights.T)  # Transpose to match PyTorch's weight shape
    bias = torch.tensor(fc_layer.bias)

    # Forward pass using the CUDA fully connected layer
    d_output = fc_layer.forward(d_input)
    output = d_output.copy_to_host()

    # Forward pass using PyTorch
    input_torch = torch.tensor(input_tensor, dtype=torch.float32)
    output_torch = torch.matmul(input_torch, weight) + bias

    # Convert PyTorch output to numpy
    output_torch_np = output_torch.detach().numpy()

    # Assert equality
    np.testing.assert_allclose(output, output_torch_np, rtol=1e-5, atol=1e-5)
    print("FullyConnectedLayer test passed!")

