class ConvLayer:
    def __init__(self, kernel_size, in_channels, out_channels, padding='same'):
        # Initialize kernels and biases
        self.kernel = ...
        self.bias = ...

    def forward(self, input_tensor):
        # Perform convolution using your Numba CUDA function
        output = convolution_layer(input_tensor, self.kernel, self.bias, padding=self.padding)
        return output

class ReLULayer:
    def forward(self, input_tensor):
        output = relu_activation(input_tensor)
        return output

class PoolingLayer:
    def __init__(self, pool_size=2, stride=2, mode='max'):
        self.pool_size = pool_size
        self.stride = stride
        self.mode = mode

    def forward(self, input_tensor):
        output = pooling_layer(input_tensor, self.pool_size, self.stride, mode=self.mode)
        return output
