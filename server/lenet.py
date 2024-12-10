from layers import ConvLayer, ReLULayer, PoolingLayer, FullyConnectedLayer, FlattenLayer, SoftmaxLayer
from numba import cuda
import numpy as np
import cv2
import json
from utils import preprocess_for_lenet

class LeNetModel:
    def __init__(self):
        self.layers = []
        self.build_model()

    def build_model(self):
        # Convolutional Layer 1
        self.layers.append(ConvLayer(1, 6, kernel_size=5, padding='valid'))  # Input channels: 1 (grayscale images), Output channels: 6
        self.layers.append(ReLULayer())
        self.layers.append(PoolingLayer(pool_size=2, stride=2))  # 2x2 max pooling

        # Convolutional Layer 2
        self.layers.append(ConvLayer(6, 16, kernel_size=5, padding='valid'))
        self.layers.append(ReLULayer())
        self.layers.append(PoolingLayer(pool_size=2, stride=2))  # 2x2 max pooling

        # Flatten Layer
        self.layers.append(FlattenLayer())

        # Fully Connected Layers
        self.layers.append(FullyConnectedLayer(16 * 5 * 5, 120))
        self.layers.append(ReLULayer())
        self.layers.append(FullyConnectedLayer(120, 84))
        self.layers.append(ReLULayer())
        self.layers.append(FullyConnectedLayer(84, 10))  # Assuming 10 classes for digits (0-9)
        self.layers.append(SoftmaxLayer())

    def load_weights(self, weight_file):
        # Load weights from file
        weights = np.load(weight_file)

        conv_idx = 0
        fc_idx = 0

        for layer in self.layers:
            if isinstance(layer, ConvLayer):
                # Load and transpose weights if necessary
                weight = weights[f'conv_{conv_idx}_weight']
                bias = weights[f'conv_{conv_idx}_bias']
                # Transpose weight dimensions
                weight = weight.transpose(2, 3, 1, 0)
                weight = np.ascontiguousarray(weight)
                # Update layer weights
                layer.kernel = weight.astype(np.float32)
                layer.bias = bias.astype(np.float32)
                # Transfer to device
                layer.d_kernel = cuda.to_device(layer.kernel)
                layer.d_bias = cuda.to_device(layer.bias)
                conv_idx += 1
            elif isinstance(layer, FullyConnectedLayer):
                weight = weights[f'fc_{fc_idx}_weight']
                bias = weights[f'fc_{fc_idx}_bias']
                # Transpose weight dimensions
                weight = weight.T
                weight = np.ascontiguousarray(weight)
                # Update layer weights
                layer.weights = weight.astype(np.float32)
                layer.bias = bias.astype(np.float32)
                # Transfer to device
                layer.d_weights = cuda.to_device(layer.weights)
                layer.d_bias = cuda.to_device(layer.bias)
                fc_idx += 1

    def forward(self, input_tensor):
        # Transfer input_tensor to device
        d_input = cuda.to_device(input_tensor.astype(np.float32))

        self.intermediate_outputs = []  # Clear outputs before each forward pass

        for layer in self.layers:
            d_input = layer.forward(d_input)
            try:
                self.intermediate_outputs.append(d_input.copy_to_host())
            except:
                self.intermediate_outputs.append(d_input)

        # Copy the final output back to host
        output = d_input  # Softmax is assumed to return output from host
        return output


if __name__ == "__main__":
    input_tensor = preprocess_for_lenet("./data/dog.jpg")

    model = LeNetModel()

    final_output = model.forward(input_tensor)
    layer_output = model.intermediate_outputs

    # Export intermediate activations to a JSON file
    export_data = [output.tolist() for output in layer_output]  # Convert to lists for JSON compatibility

    with open("./activations/lenet_layer_output.json", "w") as f:
        json.dump(export_data, f)

    print("Forward pass completed. Activations saved to './activations/lenet_layer_output.json'.")
