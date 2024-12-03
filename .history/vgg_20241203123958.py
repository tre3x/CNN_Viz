from layers import ConvLayer, ReLULayer, PoolingLayer, FullyConnectedLayer
from numba import cuda
import numpy as np
import time

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


    def forward(self, input_tensor):
        # Transfer input_tensor to device
        d_input = cuda.to_device(input_tensor.astype(np.float32))

        for layer in self.layers:
            d_input = layer.forward(d_input)

        # Copy the final output back to host
        output = d_input.copy_to_host()
        return output

    # You might also include methods for loading pretrained weights


if __name__=="__main__":
    input_tensor = np.random.rand(224, 224, 3).astype(np.float32)
    
    model = VGG16Model()
    start = time.time()
    out = model.forward(input_tensor)
    print(time.time()-start)
    print(out.shape)

