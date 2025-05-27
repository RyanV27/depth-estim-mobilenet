import os
import glob
import time
from PIL import Image
import numpy as np
import PIL
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils    
import torchvision.models as models
import cv2
import argparse
from pathlib import Path

from depth_data_mob import DepthDataset, Augmentation, ToTensor
from mobile_model import Model
# from UtilityTest import DepthDataset
# from UtilityTest import ToTensor

# Normalize images for display
def normalize_image(img):
    # if img.shape[0] == 3:  # CHW format
    img = np.transpose(img, (1, 2, 0))  # Convert to HWC
    img = (img - img.min()) / (img.max() - img.min())
    return img

def normalize_depth(depth):
    depth = depth.reshape(240,320)
    scale_percent = 200 # percent of original size
    width = int(depth.shape[1] * scale_percent / 100)
    height = int(depth.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(depth, dim, interpolation = cv2.INTER_AREA)
    return resized

def save_model_comparison(input_tensor, model, expected_depth=None, 
                         output_dir='generated_outputs', filename='comparison.png', index=0):
    """
    Save input, model output and expected output side by side as an image
    
    Args:
        input_tensor: Input tensor of shape (N, C, H, W)
        model: Your PyTorch model
        expected_output: Expected output tensor (optional)
        output_dir: Directory to save the image
        filename: Name of the output image file
        index: Which sample from the batch to visualize
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Pass through model
    with torch.no_grad():
        model_output = model(input_tensor)

    # Create figure
    fig, ax = plt.subplots(input_tensor.shape[0], 3 if expected_depth is not None else 2, figsize=(15, 20))

    for i in range(input_tensor.shape[0]):
        # Select one example to display
        np_input_img = input_tensor[i].cpu().numpy()
        np_output_depth = model_output[i].cpu().numpy()
        np_expected_depth = expected_depth[i].cpu().numpy() if expected_depth is not None else None
        
        np_input_img = normalize_image(np_input_img)
        np_output_depth = normalize_depth(np_output_depth)
        if expected_depth is not None:
            np_expected_depth = normalize_depth(np_expected_depth)

        # Display images
        ax[i, 0].imshow(np_input_img)
        ax[i, 0].axis('off')
        if i == 0:
            ax[i, 0].set_title('Input Image')
    
        ax[i, 1].imshow(np_output_depth)
        ax[i, 1].axis('off')
        if i == 0:
            ax[i, 1].set_title('Model Output')
    
        if expected_depth is not None:
            ax[i, 2].imshow(np_expected_depth)
            ax[i, 2].axis('off')
            if i == 0:
                ax[i, 2].set_title('Expected Output')

    plt.tight_layout()
    
    # Save the figure
    save_path = Path(output_dir) / filename
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory
    
    print(f"Saved comparison image to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--checkpath', type = str, required = True, help = 'Path to the checkpoint folder')
    parser.add_argument('--checkpoint', type = int, required = True, help = 'Epoch number/Checkpoint of model to test.')
    parser.add_argument('--batch', type = int, required = True, help = 'Number of images to test.')
    args = parser.parse_args()
    
    batch_size = args.batch
    
    depth_dataset_test = DepthDataset(traincsv_path='./data/nyu2_train.csv', root_dir='./',
                    transform=transforms.Compose([Augmentation(0.5),ToTensor(is_test = False)]))
    test_loader=DataLoader(depth_dataset_test, batch_size, shuffle=True)

    sample = next(iter(test_loader))

    # print(f"Images shape: {sample['image'].shape}")
    # print(f"Depths shape: {sample['depth'].shape}")

    model = Model()
    path = args.checkpath + str(args.checkpoint) + '.tar'  
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    save_model_comparison(sample['image'], model, expected_depth = sample['depth'])