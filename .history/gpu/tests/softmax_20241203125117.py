import torch
import numpy as np
from numba import cuda
from softmax_cuda import softmax

def check(computed, logits):
    # Compute softmax using PyTorch
    logits_torch = torch.tensor(logits, dtype=torch.float32)
    output_torch = torch.nn.functional.softmax(logits_torch, dim=0)
    output_torch_np = output_torch.numpy()

    # Compare CUDA and PyTorch results
    np.testing.assert_allclose(computed, output_torch_np, rtol=1e-5, atol=1e-5)
    print("Softmax: CUDA and PyTorch outputs match!")