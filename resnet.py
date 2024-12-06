from layers import ConvLayer, ReLULayer, PoolingLayer, FullyConnectedLayer, FlattenLayer, SoftmaxLayer, AddLayer
from numba import cuda
import numpy as np

class ResNetModel:
    def __init__(self):
        self.layers = []
        self.build_model()

    def residual_block(self, in_channels, out_channels, stride=1):
        """
        Residual block with two convolutional layers and a skip connection.
        Supports optional downsampling when `stride` is greater than 1.
        """
        block = []
        
        # First convolutional layer
        block.append(ConvLayer(in_channels, out_channels, kernel_size=3, stride=stride, padding='same'))
        block.append(ReLULayer())
        
        # Second convolutional layer
        block.append(ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, padding='same'))
        
        # Add skip connection
        if stride > 1 or in_channels != out_channels:
            # Downsample the input to match dimensions
            downsample = ConvLayer(in_channels, out_channels, kernel_size=1, stride=stride, padding='same')
        else:
            downsample = None

        return block, downsample

    def build_model(self):
        # Initial convolution and pooling
        self.layers.append(ConvLayer(3, 64, kernel_size=7, stride=2, padding='same'))
        self.layers.append(ReLULayer())
        self.layers.append(PoolingLayer(pool_size=3, stride=2, padding='same'))

        # ResNet Blocks
        # Block 1
        self.add_residual_block(64, 64, num_blocks=3, stride=1)
        # Block 2
        self.add_residual_block(64, 128, num_blocks=4, stride=2)
        # Block 3
        self.add_residual_block(128, 256, num_blocks=6, stride=2)
        # Block 4
        self.add_residual_block(256, 512, num_blocks=3, stride=2)

        # Classifier
        self.layers.append(PoolingLayer(pool_size=7, stride=1))  # Global average pooling
        self.layers.append(FlattenLayer())
        self.layers.append(FullyConnectedLayer(512, 1000))  # Assuming 1000 classes
        self.layers.append(SoftmaxLayer())

    def add_residual_block(self, in_channels, out_channels, num_blocks, stride):
        """
        Add a sequence of residual blocks to the model.
        """
        for i in range(num_blocks):
            block, downsample = self.residual_block(in_channels, out_channels, stride if i == 0 else 1)
            self.layers.extend(block)
            if downsample:
                self.layers.append(downsample)
            self.layers.append(AddLayer())
            self.layers.append(ReLULayer())
            in_channels = out_channels  # Update in_channels for the next block

    def load_weights(self, weight_file):
        # Load weights from file 
        weights = np.load(weight_file)
        conv_idx = 0
        fc_idx = 0

        for layer in self.layers:
            if isinstance(layer, ConvLayer):
                weight = weights[f'conv_{conv_idx}_weight']
                bias = weights[f'conv_{conv_idx}_bias']
                weight = weight.transpose(2, 3, 1, 0)
                weight = np.ascontiguousarray(weight)
                layer.kernel = weight.astype(np.float32)
                layer.bias = bias.astype(np.float32)
                layer.d_kernel = cuda.to_device(layer.kernel)
                layer.d_bias = cuda.to_device(layer.bias)
                conv_idx += 1
            elif isinstance(layer, FullyConnectedLayer):
                weight = weights[f'fc_{fc_idx}_weight']
                bias = weights[f'fc_{fc_idx}_bias']
                weight = weight.T
                weight = np.ascontiguousarray(weight)
                layer.weights = weight.astype(np.float32)
                layer.bias = bias.astype(np.float32)
                layer.d_weights = cuda.to_device(layer.weights)
                layer.d_bias = cuda.to_device(layer.bias)
                fc_idx += 1

    def forward(self, input_tensor):
        d_input = cuda.to_device(input_tensor.astype(np.float32))
        self.intermediate_outputs = []

        for layer in self.layers:
            d_input = layer.forward(d_input)
            try:
                self.intermediate_outputs.append(d_input.copy_to_host())
            except:
                self.intermediate_outputs.append(d_input)

        output = d_input
        return output


if __name__ == "__main__":
    input_tensor = preprocess_for_vgg16("./data/dog.jpg")  # Replace with a ResNet preprocessing function
    model = ResNetModel()
    model.load_weights('./weights/resnet_weights.npz')

    final_output = model.forward(input_tensor)
    layer_output = model.intermediate_outputs

    # Add appropriate tests for ResNet
    print("Final output:", final_output)
