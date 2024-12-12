from numba import cuda
import numpy as np

from gpu import convolution, pooling, relu, fcn, softmax
from cpu import convolution_cpu, pooling_cpu, relu_cpu, softmax_cpu, fcn_cpu
from tests import convolution_test

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

    def forward(self, d_input, use_gpu=True):
        # d_input is a device array
        if use_gpu:
            d_output = convolution.convolution_layer(d_input, self.d_kernel, self.d_bias, padding=self.padding)
        else:
            return convolution_cpu.convolution_layer(d_input, self.kernel, self.bias, padding=self.padding)
        return d_output
    
    def __str__(self):
            return f"ConvLayer(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, padding={self.padding})"

class ReLULayer:
    def forward(self, d_input, use_gpu=True):
        # Apply ReLU activation
        if use_gpu:
            d_output = relu.relu_activation(d_input)
        else:
            return relu_cpu.relu_activation(d_input)
        return d_output
    
    def __str__(self):
            return f"ReLULayer"
    
class SoftmaxLayer:
    def forward(self, d_input, use_gpu=True):
        if use_gpu:
            d_output = softmax.softmax_activation(d_input)
        else:
            return softmax_cpu.softmax_activation(d_input)
        return d_output
    
    def __str__(self):
            return f"SoftmaxLayer"

class PoolingLayer:
    def __init__(self, pool_size=2, stride=2, mode='max'):
        self.pool_size = pool_size
        self.stride = stride
        self.mode = mode

    def forward(self, d_input, use_gpu=True):
        if use_gpu:
            d_output = pooling.pooling_layer(d_input, self.pool_size, self.stride, mode=self.mode)
        else:
            return pooling_cpu.pooling_layer(d_input, self.pool_size, self.stride, self.mode)
        return d_output
    
    def __str__(self):
            return f"PoolingLayer"

class FlattenLayer:
    def forward(self, d_input, use_gpu=False):
        # Flatten the input tensor
        return d_input.reshape((-1,))
    
    def __str__(self):
            return f"FlattenLayer"

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

    def forward(self, d_input, use_gpu=True):
        # Perform matrix multiplication using CUDA
        if use_gpu:
            d_output = fcn.fully_connected_layer(d_input, self.d_weights, self.d_bias)
        else:
            return fcn_cpu.fully_connected_layer(d_input, self.weights, self.bias)
        return d_output
    
    def __str__(self):
            return f"FullyConnectedLayer(in_channels={self.in_features}, out_channels={self.out_features})"

# Implement the fully_connected_layer function in a similar way


if __name__=="__main__":
    inp = np.random.rand(224, 224, 3).astype(np.float32)
    obj = ConvLayer(3, 64, 3, 'same')
    out = obj.forward(inp)
    convolution_test.check(out, inp, obj.kernel, obj.bias, 'same')