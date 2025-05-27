import os
import argparse
import io
import h5py
import torch
import cv2
import numpy as np
from torchvision import transforms

from depth_data_mob import ToTensor
from mobile_model import Model

import matplotlib.pyplot as plt

def h5_loader(bytes_stream):
    # Reference: https://github.com/dwofk/fast-depth/blob/master/dataloaders/dataloader.py#L8-L13
    f = io.BytesIO(bytes_stream)
    h5f = h5py.File(f, "r")
    rgb = np.array(h5f["rgb"])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f["depth"])
    return rgb, depth

def image_transform(image):
    return ToTensor().to_tensor(image)

def depth_transform(depth):
    # depth = np.resize(depth, (240, 320))
    # depth = ToTensor().to_tensor(depth).float() * 1000
    # depth = torch.clamp(depth, 10, 1000)
    depth = cv2.resize(
        depth,
        (320, 240),  # (width, height)
        interpolation=cv2.INTER_AREA  # or INTER_CUBIC
    )
    depth = torch.from_numpy(depth)
    depth = ToTensor().to_tensor(depth) * 1000
    depth = torch.clamp(depth, 10, 1000)
    return depth

def DepthNorm(depth, maxDepth=1000.0): 
    return maxDepth / depth

def abs_rel(pred, gt, eps=1e-6):
    valid = gt > 0
    diff  = torch.abs(pred - gt) / (gt + eps)
    return diff[valid].mean()

def delta1(pred, gt, thresh=1.25, eps=1e-6):
    valid = gt > 0
    ratio = torch.max(pred/(gt+eps), gt/(pred+eps))
    return (ratio[valid] < thresh).float().mean()

def save_tensor_as_image(tensor, path, cmap='gray'):
    """Save a tensor as an image file."""
    # Remove batch dim if present and squeeze channel dim for grayscale
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # [1, C, H, W] -> [C, H, W]
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)  # [1, H, W] -> [H, W]
    
    tensor = tensor.detach().cpu().numpy()  # Convert to numpy
    plt.imsave(path, tensor, cmap=cmap)  # Save with matplotli

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--valpath', type = str, required = True, help = 'Path to the folder with validation data.')
    parser.add_argument('--checkpath', type = str, required = True, help = 'Path to the model checkpoint.')
    args = parser.parse_args()

    save_results_path = "./validation_results.txt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    checkpoint = torch.load(args.checkpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),  # Converts PIL to tensor (C x H x W) and scales to [0, 1]
    ])
    
    val_dir_path = args.valpath

    abs_rel_vals = []
    delta1_vals = []
    
    for file_name in os.listdir(val_dir_path):
        file_path = os.path.join(val_dir_path, file_name)
        print(f"\n{file_path}:")

        with open(file_path, 'rb') as file:
            bytes_stream = file.read()
            image, depth = h5_loader(bytes_stream)

            # print(f"image shape before: {image.shape}")
            # print(f"depth shape before: {depth.shape}")
            # plt.imsave('./depth_before.jpg', depth, cmap = 'viridis')

            depth = depth_transform(depth)
            depth_n = DepthNorm(depth)
            # save_tensor_as_image(depth_n, './depth_after.jpg', cmap = 'viridis')

            image = image_transform(image)
            image = image.to(device)
            output = model(image.unsqueeze(0))
            # save_tensor_as_image(output, './output.jpg', cmap = 'viridis')

            # print(f"image shape after: {image.shape}")
            # print(f"output shape: {output.shape}")
            # print(f"depth shape after: {depth.shape}")

            abs_rel_val = abs_rel(output[0, 0].to('cpu'), depth_n)
            abs_rel_vals.append(abs_rel_val.detach().numpy())
            print(f"abs_rel = {abs_rel_vals[-1]}")
            
            delta1_val = delta1(output[0, 0].to('cpu'), depth_n)
            delta1_vals.append(delta1_val.detach().numpy())
            print(f"delta1 = {delta1_vals[-1]}")

    mean_abs_rel = np.mean(abs_rel_vals)
    mean_delta1 = np.mean(delta1_vals)
    print(f"\n\nAverage Absolute Relative Error = {mean_abs_rel}")
    print(f"Average Delta1 = {mean_delta1}")

    with open(save_results_path, "a") as file:
        file.write("\nCheckpoint: " + args.checkpath)
        file.write(f"\nAverage Absolute Relative Error = {str(mean_abs_rel)}")
        file.write(f"\nAverage Delta1 = {str(mean_delta1)}")
        