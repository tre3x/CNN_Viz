import numpy as np

def pooling_layer(input_tensor, pool_size=2, stride=2, mode='max'):
    """
    Performs pooling on the CPU.

    Parameters:
    input_tensor: Input tensor (H_in, W_in, C)
    pool_size: Size of the pooling window
    stride: Stride of the pooling operation
    mode: 'max' or 'avg'

    Returns:
    output_tensor: Output tensor (H_out, W_out, C)
    """
    H_in, W_in, C = input_tensor.shape
    H_out = (H_in - pool_size) // stride + 1
    W_out = (W_in - pool_size) // stride + 1
    output_tensor = np.zeros((H_out, W_out, C), dtype=np.float32)

    for c in range(C):
        for i in range(H_out):
            for j in range(W_out):
                start_i = i * stride
                start_j = j * stride
                end_i = start_i + pool_size
                end_j = start_j + pool_size
                pool_region = input_tensor[start_i:end_i, start_j:end_j, c]
                if mode == 'max':
                    output_tensor[i, j, c] = np.max(pool_region)
                elif mode == 'avg':
                    output_tensor[i, j, c] = np.mean(pool_region)
                else:
                    raise ValueError("Invalid pooling mode. Use 'max' or 'avg'.")

    return output_tensor