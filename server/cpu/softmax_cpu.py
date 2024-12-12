import numpy as np

def softmax_activation(logits):
    """
    Performs softmax activation on the CPU.

    Parameters:
    logits: Input logits tensor

    Returns:
    probabilities: Output probabilities tensor
    """
    exp_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=0, keepdims=True)