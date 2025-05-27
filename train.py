import pandas as pd
import numpy as np
import torch
import os
import random
import cv2
import kornia
import warnings
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import time
import argparse
import datetime
import torch
import torchvision.utils as vutils 
from torchvision import transforms
from PIL import Image
from io import BytesIO
from zipfile import ZipFile
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from sklearn.utils import shuffle

from depth_data_mob import DepthDataset, Augmentation, ToTensor
from mobile_model import Model
from custom_loss_fns import CustomLosses
# from Densenet_depth_model.model_dense import Model

from torchsummary import summary

# Ignore warnings
warnings.filterwarnings("ignore")

# def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
#     ssim = kornia.losses.SSIMLoss(window_size=11,max_val=val_range,reduction='none')
#     return ssim(img1, img2)

def DepthNorm(depth, maxDepth=1000.0): 
    return maxDepth / depth

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    value = value.cpu().numpy()[0,:,:]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    #value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)

    img = value[:,:,:3]

    return img.transpose((2,0,1))

def LogProgress(model, writer, test_loader, epoch, device):
    model.eval()
    sequential = test_loader
    sample_batched = next(iter(sequential))
    image = sample_batched['image'].to(device)
    depth = sample_batched['depth'].to(device, non_blocking=True)
    if epoch == 0: 
        writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
    if epoch == 0: 
        writer.add_image('Train.2.Depth', colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)), epoch)
    output = DepthNorm( model(image) )
    writer.add_image('Train.3.Ours', colorize(vutils.make_grid(output.data, nrow=6, normalize=False)), epoch)
    writer.add_image('Train.3.Diff', colorize(vutils.make_grid(torch.abs(output-depth).data, nrow=6, normalize=False)), epoch)
    del image
    del depth
    del output

if __name__ == "__main__":
    # Arguement Parser
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--checkpath', type = str, required = True, help = 'Path to the checkpoint folder')
    parser.add_argument('--run', type = int, required = True, help = 'Current experiment number, i.e, 1 + number of times code ran before.')
    parser.add_argument('--epoch', type = int, required = True, help = 'Starting epoch of the current run. Check ./checkpoints/ for reference.')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    writer = SummaryWriter(log_dir = f'./runs/run{args.run}')
    
    if device == 'cuda' and torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
      model = torch.nn.DataParallel(model)
        
    #load trained model if needed
    #model.load_state_dict(torch.load('./1.pth'))
    print('Model created.')
    
    epochs = 50
    lr = 0.0001
    batch_size = 64     # 64
    
    depth_dataset_train = DepthDataset(traincsv_path='./data/nyu2_train.csv', root_dir='./',
                    transform=transforms.Compose([Augmentation(0.5),ToTensor()]))
    train_loader=DataLoader(depth_dataset_train, batch_size, shuffle=True)
    depth_dataset_test = DepthDataset(traincsv_path='./data/nyu2_test.csv', root_dir='./',
                    transform=transforms.Compose([Augmentation(0.5),ToTensor()]))
    test_loader=DataLoader(depth_dataset_test, batch_size, shuffle=True)
    
    # l1_criterion = torch.nn.L1Loss()
    criterion = CustomLosses()
    
    optimizer = torch.optim.Adam( model.parameters(), lr )

    start_epoch = args.epoch
    # Start training...
    for epoch in range(start_epoch, epochs):
        if epoch > 0:
            path = args.checkpath + str(epoch - 1) + '.tar'        
            # torch.save(model.state_dict(), path)
            checkpoint = torch.load(path, weights_only = True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded epoch {epoch - 1} model and optimizer state_dict.")
        
        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(train_loader)
    
        # Switch to train mode
        model.train()
    
        end = time.time()
    
        for i, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()
    
            #Prepare sample and target
            image = sample_batched['image'].to(device)
            depth = sample_batched['depth'].to(device, non_blocking=True)
                
            # Normalize depth
            depth_n = DepthNorm( depth )
            
            # Predict
            # output = model(image[:, :, :432, :432])   
            # output = model(image[:, :, :240, :320])   
            output = model(image)
            
            # Compute the loss
            # l_depth = l1_criterion(output, depth_n)
            # l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)
    
            # loss = (1.0 * l_ssim.mean().item()) + (0.1 * l_depth)
            loss = criterion.ssim_l1_edge_criterion(output, depth_n)              # Missed to pass normalized depth (Try run 4)          
            
            # Update step
            losses.update(loss.data.item(), image.size(0))
            
            loss.backward()
            optimizer.step()
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))
            
            # Log progress
            niter = epoch*N+i
            if i % 5 == 0:
                # Print to console
                print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                'ETA {eta}\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'
                .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))
    
                # Log to tensorboard
                writer.add_scalar('Train/Loss', losses.val, niter)

            if i % 300 == 0:
                LogProgress(model, writer, test_loader, niter, device)

        if epoch % 1 == 0:
            path = args.checkpath + str(epoch) + '.tar'           
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, path)
            print(f"Saved epoch {epoch} model and optimizer state_dict.")
    
            # Record epoch's intermediate results
            LogProgress(model, writer, test_loader, niter, device)
            writer.add_scalar('Train/Loss.avg', losses.avg, epoch)

    writer.close()

    # #Evaluations
    # model = Model().cuda()
    # model = torch.nn.DataParallel(model)
    # #load the model if needed
    # #model.load_state_dict(torch.load('./3.pth'))
    # model.eval()
    # batch_size=1
    
    # depth_dataset = DepthDataset(traincsv=traincsv, root_dir='/workspace/',
    #                 transform=transforms.Compose([Augmentation(0.5),ToTensor()]))
    # train_loader=DataLoader(depth_dataset, batch_size, shuffle=True)
    
    # for sample_batched1  in (train_loader):
    #     image1 = torch.autograd.Variable(sample_batched1['image'].cuda())
        
    #     outtt=model(image1 )
    #     break