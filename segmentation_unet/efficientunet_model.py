"""
EfficientUNet model for HC18 segmentation
Based on models1.ipynb architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class EfficientUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # --------- ENCODER (EfficientNet-B0) ---------
        self.encoder = EfficientNet.from_pretrained("efficientnet-b0")

        self.enc0 = nn.Sequential(self.encoder._conv_stem, self.encoder._bn0)
        self.enc1 = self.encoder._blocks[0:2]      # 24 ch
        self.enc2 = self.encoder._blocks[2:4]      # 40 ch
        self.enc3 = self.encoder._blocks[4:10]     # 112 ch
        self.enc4 = self.encoder._blocks[10:]      # 320 ch (deepest)

        # Channel sizes from EfficientNet-B0
        c0 = 32     # stem output
        c1 = 24
        c2 = 40
        c3 = 112
        c4 = 320

        # --------- DECODER ---------
        self.up4 = self.up_block(c4, c3)
        self.up3 = self.up_block(c3 + c3, c2)
        self.up2 = self.up_block(c2 + c2, c1)
        self.up1 = self.up_block(c1 + c1, c0)

        # Final output: concatenation => 32 + 32 = 64 channels
        self.final = nn.Conv2d(c0 * 2, 1, kernel_size=1)

    def run_block(self, block_list, x):
        """EfficientNet blocks are in a ModuleList, so run manually."""
        for block in block_list:
            x = block(x)
        return x

    def up_block(self, in_c, out_c):
        """Decoder upsampling block."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        # --------- ENCODER ---------
        e0 = self.enc0(x)                  # 256 → 128
        e1 = self.run_block(self.enc1, e0) # 128 → 64
        e2 = self.run_block(self.enc2, e1) # 64 → 32
        e3 = self.run_block(self.enc3, e2) # 32 → 16
        e4 = self.run_block(self.enc4, e3) # 16 → 8

        # --------- DECODER ---------
        d4 = self.up4(e4)                  # 8 → 16
        d4 = torch.cat([d4, e3], dim=1)

        d3 = self.up3(d4)                  # 16 → 32
        d3 = torch.cat([d3, e2], dim=1)

        d2 = self.up2(d3)                  # 32 → 64
        d2 = torch.cat([d2, e1], dim=1)

        d1 = self.up1(d2)                  # 64 → 128
        d1 = torch.cat([d1, e0], dim=1)

        out = self.final(d1)               # 128×128 output

        # --------- FINAL UPSAMPLING TO 256×256 ---------
        out = F.interpolate(out, size=(256, 256),
                            mode='bilinear', align_corners=False)

        return torch.sigmoid(out)

class DiceBCELoss(nn.Module):
    """Combined Dice + Binary Cross Entropy loss"""
    def __init__(self, dice_weight=0.7, bce_weight=0.3):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCELoss()
    
    def dice_loss(self, pred, target, smooth=1e-6):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return 1 - dice
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce(pred, target)
        
        return self.dice_weight * dice + self.bce_weight * bce


class FocalLoss(nn.Module):
    """Focal Loss for extreme class imbalance - focuses on hard examples"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Weight for positive class (foreground)
        self.gamma = gamma  # Focusing parameter (higher = more focus on hard examples)
    
    def forward(self, pred, target):
        # pred: sigmoid outputs [B, 1, H, W]
        # target: binary mask [B, 1, H, W]
        
        # Binary cross entropy
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Focal term: (1 - p_t)^gamma
        p_t = pred * target + (1 - pred) * (1 - target)  # Probability of correct class
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting for positive/negative samples
        if self.alpha is not None:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_weight = alpha_t * focal_weight
        
        focal_loss = focal_weight * bce_loss
        return focal_loss.mean()


class DiceFocalLoss(nn.Module):
    """Combined Dice + Focal Loss - best for extreme class imbalance like HC segmentation"""
    def __init__(self, dice_weight=0.7, focal_weight=0.3, focal_alpha=0.75, focal_gamma=2.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def dice_loss(self, pred, target, smooth=1e-6):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return 1 - dice
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal(pred, target)
        
        return self.dice_weight * dice + self.focal_weight * focal


class TverskyLoss(nn.Module):
    """Tversky Loss - better for extremely small targets"""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # False positive weight (low = less penalty for FP)
        self.beta = beta    # False negative weight (high = more penalty for FN)
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        TP = (pred * target).sum()
        FP = (pred * (1 - target)).sum()
        FN = ((1 - pred) * target).sum()
        
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky_index


class DiceTverskyLoss(nn.Module):
    """Combined Dice + Tversky for extreme small targets"""
    def __init__(self, dice_weight=0.5, tversky_weight=0.5, tversky_alpha=0.3, tversky_beta=0.7):
        super().__init__()
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
    
    def dice_loss(self, pred, target, smooth=1e-6):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return 1 - dice
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        tversky = self.tversky(pred, target)
        
        return self.dice_weight * dice + self.tversky_weight * tversky

def dice_coefficient(pred, target, threshold=0.5, smooth=1e-6):
    """Calculate Dice coefficient for evaluation"""
    pred = (pred > threshold).float()
    target = (target > 0.5).float()
    
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()

def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    """Calculate IoU (Jaccard) score"""
    pred = (pred > threshold).float()
    target = (target > 0.5).float()
    
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def pixel_accuracy(pred, target, threshold=0.5):
    """Calculate pixel-wise accuracy"""
    pred = (pred > threshold).float()
    target = (target > 0.5).float()
    correct = (pred == target).float().sum()
    total = torch.numel(target)
    return (correct / total).item()
