"""
U-Net model with FetalCLIP encoder for head circumference segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Double convolution block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNetDecoder(nn.Module):
    """U-Net decoder with skip connections"""
    def __init__(self, encoder_channels=[256, 256, 256, 256], decoder_channels=[256, 128, 64, 32]):
        super().__init__()
        
        # Decoder blocks (no skip connections since ViT features are all same resolution)
        self.up1 = nn.ConvTranspose2d(encoder_channels[0], decoder_channels[0], 2, stride=2)
        self.conv1 = ConvBlock(decoder_channels[0], decoder_channels[0])
        
        self.up2 = nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], 2, stride=2)
        self.conv2 = ConvBlock(decoder_channels[1], decoder_channels[1])
        
        self.up3 = nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2], 2, stride=2)
        self.conv3 = ConvBlock(decoder_channels[2], decoder_channels[2])
        
        self.up4 = nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3], 2, stride=2)
        self.conv4 = ConvBlock(decoder_channels[3], decoder_channels[3])
        
        # Final output
        self.final = nn.Conv2d(decoder_channels[3], 1, 1)
    
    def forward(self, x):
        """
        Args:
            x: encoder output [B, C, H, W]
        """
        # Upsampling path (no skip connections for ViT)
        x = self.up1(x)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = self.conv4(x)
        
        # Final segmentation map
        x = self.final(x)
        return x

class FetalCLIPUNet(nn.Module):
    """U-Net with FetalCLIP Vision Transformer encoder"""
    def __init__(self, fetalclip_model, decoder_channels=[256, 128, 64, 32], freeze_encoder=True):
        super().__init__()
        
        self.encoder = fetalclip_model.visual
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Projection to convert transformer output to spatial features
        embed_dim = self.encoder.transformer.width  # 1024 for FetalCLIP
        self.proj_to_spatial = nn.Conv2d(embed_dim, decoder_channels[0], 1)
        
        # U-Net decoder
        encoder_channels = [decoder_channels[0], decoder_channels[0], decoder_channels[0], decoder_channels[0]]
        self.decoder = UNetDecoder(encoder_channels, decoder_channels)
    
    def extract_features_from_transformer(self, x):
        """Extract multi-level features from ViT encoder"""
        B = x.shape[0]
        
        # Patch embedding
        x = self.encoder.conv1(x)  # [B, embed_dim, H/14, W/14]
        grid_size = x.shape[-1]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, embed_dim, H*W]
        x = x.permute(0, 2, 1)  # [B, H*W, embed_dim]
        
        # Add positional embedding and class token
        x = torch.cat([self.encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.encoder.positional_embedding.to(x.dtype)
        x = self.encoder.ln_pre(x)
        
        # Extract features from different transformer blocks
        x = x.permute(1, 0, 2)  # [seq_len, B, embed_dim]
        
        features = []
        num_blocks = len(self.encoder.transformer.resblocks)
        
        for idx, block in enumerate(self.encoder.transformer.resblocks):
            x = block(x)
            # Extract features from 25%, 50%, 75%, 100% of blocks
            if idx in [num_blocks // 4, num_blocks // 2, 3 * num_blocks // 4, num_blocks - 1]:
                # Remove class token and reshape to spatial
                feat = x[1:, :, :].permute(1, 0, 2)  # [B, H*W, embed_dim]
                feat = feat.reshape(B, grid_size, grid_size, -1)
                feat = feat.permute(0, 3, 1, 2)  # [B, embed_dim, H, W]
                features.append(feat)
        
        return features
    
    def forward(self, x):
        """
        Args:
            x: input image [B, 3, 256, 256]
        Returns:
            segmentation mask [B, 1, 256, 256]
        """
        # Extract features from encoder (use last layer)
        encoder_features = self.extract_features_from_transformer(x)
        
        # Project last feature to decoder input
        last_feat = self.proj_to_spatial(encoder_features[-1])
        
        # Decode
        mask = self.decoder(last_feat)
        
        # Upsample to original size if needed
        mask = F.interpolate(mask, size=(256, 256), mode='bilinear', align_corners=False)
        
        return torch.sigmoid(mask)

class DiceBCELoss(nn.Module):
    """Combined Dice + Binary Cross Entropy loss"""
    def __init__(self, dice_weight=0.7, bce_weight=0.3):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCELoss()
    
    def dice_loss(self, pred, target, smooth=1e-6):
        """Dice loss for binary segmentation"""
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return 1 - dice
    
    def forward(self, pred, target):
        """
        Args:
            pred: predicted mask [B, 1, H, W]
            target: ground truth mask [B, 1, H, W]
        """
        dice = self.dice_loss(pred, target)
        bce = self.bce(pred, target)
        
        return self.dice_weight * dice + self.bce_weight * bce

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
