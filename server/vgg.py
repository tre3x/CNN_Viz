from layers import ConvLayer, ReLULayer, PoolingLayer, FullyConnectedLayer, FlattenLayer, SoftmaxLayer
from numba import cuda
import numpy as np
import time
import cv2
import numpy as np
from tests import vgg_test, vgg_weights_test
from utils import preprocess_for_vgg16
import json

class VGG16Model:
    def __init__(self):
        self.layers = []
        self.build_model()

    def build_model(self):
        # Block 1
        self.layers.append(ConvLayer(3, 64, kernel_size=3, padding='same'))
        self.layers.append(ReLULayer())
        self.layers.append(ConvLayer(64, 64, kernel_size=3, padding='same'))
        self.layers.append(ReLULayer())
        self.layers.append(PoolingLayer())

        # Block 2
        self.layers.append(ConvLayer(64, 128, kernel_size=3, padding='same'))
        self.layers.append(ReLULayer())
        self.layers.append(ConvLayer(128, 128, kernel_size=3, padding='same'))
        self.layers.append(ReLULayer())
        self.layers.append(PoolingLayer())

        # Block 3
        self.layers.append(ConvLayer(128, 256, kernel_size=3, padding='same'))
        self.layers.append(ReLULayer())
        self.layers.append(ConvLayer(256, 256, kernel_size=3, padding='same'))
        self.layers.append(ReLULayer())
        self.layers.append(ConvLayer(256, 256, kernel_size=3, padding='same'))
        self.layers.append(ReLULayer())
        self.layers.append(PoolingLayer())

        # Block 4
        self.layers.append(ConvLayer(256, 512, kernel_size=3, padding='same'))
        self.layers.append(ReLULayer())
        self.layers.append(ConvLayer(512, 512, kernel_size=3, padding='same'))
        self.layers.append(ReLULayer())
        self.layers.append(ConvLayer(512, 512, kernel_size=3, padding='same'))
        self.layers.append(ReLULayer())
        self.layers.append(PoolingLayer())

        # Block 5
        self.layers.append(ConvLayer(512, 512, kernel_size=3, padding='same'))
        self.layers.append(ReLULayer())
        self.layers.append(ConvLayer(512, 512, kernel_size=3, padding='same'))
        self.layers.append(ReLULayer())
        self.layers.append(ConvLayer(512, 512, kernel_size=3, padding='same'))
        self.layers.append(ReLULayer())
        self.layers.append(PoolingLayer())


        # Classifier
        self.layers.append(FlattenLayer())
        self.layers.append(FullyConnectedLayer(512 * 7 * 7, 4096))
        self.layers.append(ReLULayer())
        self.layers.append(FullyConnectedLayer(4096, 4096))
        self.layers.append(ReLULayer())
        self.layers.append(FullyConnectedLayer(4096, 1000))  # Assuming 1000 classes
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
        #output = d_input.copy_to_host()
        output = d_input #currently softmax is returning output from host; need to fix it and return the output from device
        return output


if __name__=="__main__":
    input_tensor = preprocess_for_vgg16("./data/dog.jpg")
    
    model = VGG16Model()
    model.load_weights('./weights/vgg16_weights.npz')
    #vgg_weights_test.compare_weights(model)

    final_output = model.forward(input_tensor)
    layer_output = model.intermediate_outputs

    # Assuming layer_output is a list of NumPy arrays
    export_data = [output.tolist() for output in layer_output]  # Convert to lists for JSON compatibility

    with open("./activations/vgg_layer_output.json", "w") as f:
        json.dump(export_data, f)
    
    #vgg_test.check(final_output, input_tensor, layer_output)


