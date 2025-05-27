import torch
import torch.nn.functional as F
import kornia.losses            # Install with : pip install kornia
import math

class CustomLosses():
    def __init__(self):
        self.L1loss = torch.nn.L1Loss()
        
    # def SSIMLoss(predictions, targets, C1=0.01**2, C2=0.03**2):
    #     # Compute mean and variance for predictions and targets
    #     mu_x = F.avg_pool2d(predictions, 3, 1)
    #     mu_y = F.avg_pool2d(targets, 3, 1)
    #     sigma_x = F.avg_pool2d(predictions ** 2, 3, 1) - mu_x ** 2
    #     sigma_y = F.avg_pool2d(targets ** 2, 3, 1) - mu_y ** 2
    #     sigma_xy = F.avg_pool2d(predictions * targets, 3, 1) - mu_x * mu_y
    
    #     # Compute SSIM score
    #     ssim_numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    #     ssim_denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    #     ssim_score = ssim_numerator / ssim_denominator
    #     return 1 - ssim_score.mean()

    # def L1Loss(self, y_pred, y_true):
        # return(torch.mean(torch.abs(y_true - y_pred)))

    def SSIMLoss(self, img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
        ssim = kornia.losses.SSIMLoss(window_size=11,max_val=val_range,reduction='none')
        return ssim(img1, img2)
    
    def image_gradients(self, image):
        """Returns image gradients (dy, dx) for each color channel.
    
        Args:
            image: Tensor with shape [batch_size, h, w, d] (NHWC format)
    
        Returns:
            Pair of tensors (dy, dx) holding the vertical and horizontal image
            gradients (1-step finite difference).
    
        Raises:
            ValueError: If `image` is not a 4D tensor.
        """
        if image.dim() != 4:
            raise ValueError('image_gradients expects a 4D tensor '
                           '[batch_size, h, w, d], not {}.'.format(image.shape))
    
        # Create zero padding for last row (dy) and last column (dx)
        batch_size, height, width, depth = image.shape
    
        # Calculate dy (vertical gradients: I(x,y+1) - I(x,y))
        dy = image[:, 1:, :, :] - image[:, :-1, :, :]
        # Pad with zeros at the bottom
        dy = F.pad(dy, (0, 0, 0, 0, 0, 1), mode='constant', value=0)
    
        # Calculate dx (horizontal gradients: I(x+1,y) - I(x,y))
        dx = image[:, :, 1:, :] - image[:, :, :-1, :]
        # Pad with zeros at the right
        dx = F.pad(dx, (0, 0, 0, 1, 0, 0), mode='constant', value=0)
    
        return dy, dx
    
    
    def EdgeLoss(self, y_pred, y_true):
        """
        Compute edge-aware loss between y_true and y_pred.
        Inputs:
            y_true: Ground truth (B, C, H, W)
            y_pred: Predicted image (B, C, H, W)
        Returns:
            edges_loss: Scalar tensor
        """
        # Compute gradients (edges)
        dy_true, dx_true = self.image_gradients(y_true.permute(0, 2, 3, 1))
        dy_true = dy_true.permute(0, 3, 1, 2)
        dx_true = dx_true.permute(0, 3, 1, 2)
    
        dy_pred, dx_pred = self.image_gradients(y_pred.permute(0, 2, 3, 1))
        dy_pred = dy_pred.permute(0, 3, 1, 2)
        dx_pred = dx_pred.permute(0, 3, 1, 2)
    
        # Dynamic edge weights (exp of mean absolute gradients)
        weights_x = torch.exp(torch.mean(torch.abs(dx_true)))
        weights_y = torch.exp(torch.mean(torch.abs(dy_true)))
    
        # Edge-weighted smoothness
        smoothness_x = dx_pred * weights_x
        smoothness_y = dy_pred * weights_y
    
        # Sum mean absolute errors
        edges_loss = torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))
    
        return edges_loss
    
    # Custom Loss 1
    def ssim_l1_edge_criterion(self, y_pred, y_true):
        # Key Insights
        # SSIM Loss: Focuses on perceptual quality (structural similarity).
        # L1 Loss: Ensures pixel-wise correctness.
        # Edge Loss: Encourages the model to preserve fine details (e.g., textures, boundaries).
        # Weighted Sum: Allows tuning the trade-off between different objectives.
    
        # Define Weights for each Losses
        w_ssim = 1.0
        w_l1 = 0.1       # run 1 : 0.1, run 2 : 0.01, run 3 : 0.1
        w_edges = 0.0    # run 1 : 0.2, run 2 : 0.02, run 3: 0.0
    
        # Structural Similarity Index (SSIM) Loss
        # ssim_loss = self.SSIMLoss(y_pred, y_true)
        # ssim_loss = kornia.losses.ssim_loss(y_true, y_pred, window_size=7, max_val=640.0)
        ssim_loss = torch.clamp((1 - self.SSIMLoss(y_pred, y_true, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)
        ssim_loss = ssim_loss.mean().item()
        # print(f"SSIM loss = {ssim_loss}")
    
        # L1 Loss
        # l1_loss = L1Loss(y_pred, y_true)
        l1_loss = self.L1loss(y_pred, y_true)
        # print(f"L1 loss = {l1_loss}")
    
        # Edge loss
        edges_loss = self.EdgeLoss(y_pred, y_true) if (w_edges != 0.0) else 0.0
        # edges_loss = edge_loss(y_true, y_pred)
        # print(f"Edges loss = {edges_loss}")
    
        # Final Loss
        loss = (ssim_loss * w_ssim) + (l1_loss * w_l1) + (edges_loss * w_edges)
    
        return loss
    
    # Custom Loss 2
    def scale_invariant_criterion(self, y_pred, y_true, lamda=0.5):
        h, w = y_true.shape[-2], y_true.shape[-1]
    
        d = y_pred - y_true
        d = d.flatten(start_dim=2)
    
        square_d = d ** 2
        sum_square_d = torch.sum(square_d, dim=1)
        sum_d = torch.sum(d, dim=1)
        square_sum_d = sum_d ** 2
    
        loss = torch.mean((sum_square_d / h*w) - (lamda * square_sum_d / math.pow(h*w, 2)))
    
        return loss