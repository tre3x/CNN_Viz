import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

def check(computed_output, input_tensor, intermediate_outputs_cuda):
    # Prepare the input for the Torch model
    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC to CHW
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    
    # Load the Torch VGG model
    vgg16_model = models.vgg16(pretrained=True)
    vgg16_model.eval()

    with torch.no_grad():
        torch_output = vgg16_model(input_tensor)
        
    torch_output = torch_output.squeeze()

    np.testing.assert_allclose(computed_output, torch_output, rtol=1e-5, atol=1e-5)
    print("Final output: CUDA and PyTorch outputs match!")


