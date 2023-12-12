import torch
import torch.nn as nn

def weighted_mse_loss(pred, target, weight=1.0, threshold=50.0):
    """
    Custom weighted MSE loss which applies a weight to errors on target values above a threshold.

    Args:
    - pred (torch.Tensor): Predicted values.
    - target (torch.Tensor): Ground truth values.
    - weight (float): Weight to apply to the loss for values above the threshold.
    - threshold (float): Threshold above which the weight is applied.

    Returns:
    - torch.Tensor: Computed weighted MSE loss.
    """
    mse_loss = nn.MSELoss()(pred, target)
    mask = target > threshold
    weighted_loss = mse_loss * (weight * mask.float() + (1.0 - mask.float()))
    return torch.mean(weighted_loss)

def dice_loss(pred, target, threshold=50):
    """
    Dice loss computation for binary classification problems.

    Args:
    - pred (torch.Tensor): Predicted values.
    - target (torch.Tensor): Ground truth values.
    - threshold (float): Threshold for binarization of predictions and targets.

    Returns:
    - torch.Tensor: Computed Dice loss.
    """
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    intersection = (pred * target).sum()
    dice_coef = (2. * intersection) / (pred.sum() + target.sum() + 1e-6)
    return 1 - dice_coef

def new_combined_loss(pred, target, alpha, beta, gamma):
    """
    Combined loss function using a mix of MSE, weighted MSE, and Dice loss.

    Args:
    - pred (torch.Tensor): Predicted values.
    - target (torch.Tensor): Ground truth values.
    - alpha (float): Weight for MSE loss.
    - beta (float): Weight for weighted MSE loss.
    - gamma (float): Weight for Dice loss.

    Returns:
    - torch.Tensor: Computed combined loss.
    """
    mse = nn.MSELoss()(pred, target)
    wmse = weighted_mse_loss(pred, target)
    dice = dice_loss(pred, target)
    return alpha * mse + beta * wmse + gamma * dice


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

    
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out,size,stride,padding):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(size=size),
            nn.Conv3d(ch_in,ch_out,kernel_size=3,stride=stride,padding=padding,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
