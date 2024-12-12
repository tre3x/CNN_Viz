import numpy as np

def relu_activation(input_tensor):
    """
    Performs ReLU activation on the CPU.

    Parameters:
    input_tensor: Input tensor

    Returns:
    output_tensor: Output tensor after applying ReLU
    """
    return np.maximum(0, input_tensor)