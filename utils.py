import torch
import numpy as np
from torchvision import models
from torchvision import transforms
from PIL import Image


def vgg16_save_pretrained_weights():
    # Load the pretrained VGG16 model
    vgg16 = models.vgg16(pretrained=True)
    vgg16.eval()  # Set to evaluation mode

    # Dictionary to hold the weights and biases
    weights_dict = {}

    # Index to keep track of convolutional layers in your model
    conv_idx = 0

    # Iterate over the features (convolutional layers)
    for module in vgg16.features:
        if isinstance(module, torch.nn.Conv2d):
            # Get weights and biases
            weight = module.weight.data.numpy()
            bias = module.bias.data.numpy()

            # Save to weights_dict with continuous indices
            weights_dict[f'conv_{conv_idx}_weight'] = weight
            weights_dict[f'conv_{conv_idx}_bias'] = bias

            conv_idx += 1

    # Index for fully connected layers
    fc_idx = 0

    # Iterate over the classifier (fully connected layers)
    for module in vgg16.classifier:
        if isinstance(module, torch.nn.Linear):
            # Get weights and biases
            weight = module.weight.data.numpy()
            bias = module.bias.data.numpy()

            # Save to weights_dict
            weights_dict[f'fc_{fc_idx}_weight'] = weight
            weights_dict[f'fc_{fc_idx}_bias'] = bias

            fc_idx += 1

    np.savez('./weights/vgg16_weights.npz', **weights_dict)
    
    
def preprocess_for_vgg16(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224
        transforms.ToTensor(),          # Convert image to Tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Mean for ImageNet dataset
            std=[0.229, 0.224, 0.225]    # Standard deviation for ImageNet dataset
        )
    ])
    
    image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
    
    image_tensor = preprocess(image)
    
    # Add a batch dimension (N, C, H, W)
    image_tensor = image_tensor.unsqueeze(0)
    image_np = image_tensor.cpu().detach().numpy()
    input_tensor = np.squeeze(image_np, axis=0)
    input_tensor = np.transpose(input_tensor, (1, 2, 0))
    input_array = np.ascontiguousarray(input_tensor)

    return input_array

if __name__ == '__main__':
    vgg16_save_pretrained_weights()
