# layers.py

from numba import cuda
import numpy as np

# Import your previously defined CUDA functions
from convolution_cuda import convolution_layer
from relu_cuda import relu_activation
from pooling_cuda import pooling_layer

class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, padding='same'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding = padding

        # Initialize weights and biases
        self.kernel = np.random.randn(
            self.kernel_size[0], self.kernel_size[1], in_channels, out_channels
        ).astype(np.float32) * np.sqrt(2. / in_channels)
        self.bias = np.zeros(out_channels, dtype=np.float32)

        # Transfer weights and biases to device memory
        self.d_kernel = cuda.to_device(self.kernel)
        self.d_bias = cuda.to_device(self.bias)

    def forward(self, d_input):
        # d_input is a device array
        d_output = convolution_layer(d_input, self.d_kernel, self.d_bias, padding=self.padding)
        return d_output

class ReLULayer:
    def forward(self, d_input):
        # Apply ReLU activation
        d_output = relu_activation(d_input)
        return d_output

class PoolingLayer:
    def __init__(self, pool_size=2, stride=2, mode='max'):
        self.pool_size = pool_size
        self.stride = stride
        self.mode = mode

    def forward(self, d_input):
        d_output = pooling_layer(d_input, self.pool_size, self.stride, mode=self.mode)
        return d_output

class FlattenLayer:
    def forward(self, d_input):
        # Flatten the input tensor
        return d_input.reshape((-1,))

class FullyConnectedLayer:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights and biases
        self.weights = np.random.randn(in_features, out_features).astype(np.float32) * np.sqrt(2. / in_features)
        self.bias = np.zeros(out_features, dtype=np.float32)

        # Transfer weights and biases to device memory
        self.d_weights = cuda.to_device(self.weights)
        self.d_bias = cuda.to_device(self.bias)

    def forward(self, d_input):
        # Perform matrix multiplication using CUDA
        d_output = fully_connected_layer(d_input, self.d_weights, self.d_bias)
        return d_output

# Implement the fully_connected_layer function in a similar way
