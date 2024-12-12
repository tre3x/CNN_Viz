import numpy as np

def fully_connected_layer(input_tensor, weights, bias):
    """
    Performs fully connected layer operation on the CPU.

    Parameters:
    input_tensor: Input tensor (N_in,)
    weights: Weights matrix (N_in, N_out)
    bias: Bias vector (N_out,)

    Returns:
    output_tensor: Output tensor (N_out,)
    """
    return np.dot(input_tensor, weights) + bias