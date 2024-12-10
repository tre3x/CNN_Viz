# compare_weights.py

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from vgg import VGG16Model

def compare_weights(cuda_model):
    """
    Compares the weights and biases of the custom CUDA VGG16 model with PyTorch's pretrained VGG16 model.

    Parameters:
    - cuda_model: The custom Numba CUDA VGG16 model instance.

    Returns:
    - None. Prints the comparison results layer by layer.
    """
    # Initialize PyTorch's pretrained VGG16 model
    torch_model = models.vgg16(pretrained=True)
    torch_model.eval()

    # Lists to store weights and biases from both models
    torch_weights = []
    torch_biases = []
    cuda_weights = []
    cuda_biases = []

    # Extract weights and biases from PyTorch model
    # For convolutional layers
    for layer in torch_model.features:
        if isinstance(layer, nn.Conv2d):
            weight = layer.weight.data.cpu().numpy()
            bias = layer.bias.data.cpu().numpy()
            torch_weights.append(weight)
            torch_biases.append(bias)
    # For fully connected layers
    for layer in torch_model.classifier:
        if isinstance(layer, nn.Linear):
            weight = layer.weight.data.cpu().numpy()
            bias = layer.bias.data.cpu().numpy()
            torch_weights.append(weight)
            torch_biases.append(bias)

    # Extract weights and biases from CUDA model
    for layer in cuda_model.layers:
        if isinstance(layer, ConvLayer):
            # Transpose the kernel back to match PyTorch's format
            weight = layer.kernel.transpose(3, 2, 0, 1)
            bias = layer.bias
            cuda_weights.append(weight)
            cuda_biases.append(bias)
        elif isinstance(layer, FullyConnectedLayer):
            # Transpose weights back to match PyTorch's format
            weight = layer.weights.T
            bias = layer.bias
            cuda_weights.append(weight)
            cuda_biases.append(bias)

    # Compare weights and biases layer by layer
    for idx, (cw, tw) in enumerate(zip(cuda_weights, torch_weights)):
        print(f"Comparing weights of layer {idx}:")
        try:
            np.testing.assert_allclose(cw, tw, rtol=1e-5, atol=1e-5)
            print(f"Layer {idx}: Weights match!")
        except AssertionError as e:
            print(f"Layer {idx}: Weights do not match!")
            print(f"Max absolute difference: {np.max(np.abs(cw - tw))}")
            print(f"Max relative difference: {np.max(np.abs(cw - tw) / np.maximum(np.abs(tw), 1e-8))}")
            # Optionally, print more details or break the loop
            # break

    for idx, (cb, tb) in enumerate(zip(cuda_biases, torch_biases)):
        print(f"Comparing biases of layer {idx}:")
        try:
            np.testing.assert_allclose(cb, tb, rtol=1e-5, atol=1e-5)
            print(f"Layer {idx}: Biases match!")
        except AssertionError as e:
            print(f"Layer {idx}: Biases do not match!")
            print(f"Max absolute difference: {np.max(np.abs(cb - tb))}")
            print(f"Max relative difference: {np.max(np.abs(cb - tb) / np.maximum(np.abs(tb), 1e-8))}")
            # Optionally, print more details or break the loop
            # break

if __name__ == "__main__":
    # Initialize your custom VGG16 model
    model = VGG16Model()
    # Load pretrained weights into your model
    model.load_weights('./weights/vgg16_weights.npz')
    
    # Compare weights with PyTorch's model
    compare_weights(model)