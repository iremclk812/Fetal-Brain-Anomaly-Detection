"""
Attention U-Net with EfficientNet-B0 encoder
Attention gates help focus on small targets like HC
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class AttentionGate(nn.Module):
    """Attention Gate for focusing on relevant features"""
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        
        # Gate signal from decoder
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        # Skip connection from encoder
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        # Output attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, gate, skip):
        """
        gate: decoder feature (upsampled)
        skip: encoder feature (skip connection)
        """
        # Resize gate to match skip if needed
        if gate.shape[2:] != skip.shape[2:]:
            gate = F.interpolate(gate, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # Attention coefficients
        gate_features = self.W_gate(gate)
        skip_features = self.W_skip(skip)
        
        attention = self.relu(gate_features + skip_features)
        attention_coeffs = self.psi(attention)
        
        # Apply attention to skip connection
        return skip * attention_coeffs


class AttentionUNet(nn.Module):
    """Attention U-Net with EfficientNet-B0 encoder"""
    def __init__(self):
        super().__init__()

        # --------- ENCODER (EfficientNet-B0) ---------
        self.encoder = EfficientNet.from_pretrained("efficientnet-b0")

        self.enc0 = nn.Sequential(self.encoder._conv_stem, self.encoder._bn0)
        self.enc1 = self.encoder._blocks[0:2]      # 24 ch
        self.enc2 = self.encoder._blocks[2:4]      # 40 ch
        self.enc3 = self.encoder._blocks[4:10]     # 112 ch
        self.enc4 = self.encoder._blocks[10:]      # 320 ch (deepest)

        # Channel sizes
        c0 = 32
        c1 = 24
        c2 = 40
        c3 = 112
        c4 = 320

        # --------- ATTENTION GATES ---------
        self.att4 = AttentionGate(gate_channels=c3, skip_channels=c3, inter_channels=c3//2)
        self.att3 = AttentionGate(gate_channels=c2, skip_channels=c2, inter_channels=c2//2)
        self.att2 = AttentionGate(gate_channels=c1, skip_channels=c1, inter_channels=c1//2)
        self.att1 = AttentionGate(gate_channels=c0, skip_channels=c0, inter_channels=c0//2)

        # --------- DECODER ---------
        self.up4 = self.up_block(c4, c3)
        self.up3 = self.up_block(c3 + c3, c2)
        self.up2 = self.up_block(c2 + c2, c1)
        self.up1 = self.up_block(c1 + c1, c0)

        # Final output
        self.final = nn.Conv2d(c0 * 2, 1, kernel_size=1)

    def run_block(self, block_list, x):
        """EfficientNet blocks are in a ModuleList"""
        for block in block_list:
            x = block(x)
        return x

    def up_block(self, in_c, out_c):
        """Decoder upsampling block"""
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

        # --------- DECODER WITH ATTENTION ---------
        d4 = self.up4(e4)                       # 8 → 16
        e3_att = self.att4(gate=d4, skip=e3)    # Attention on e3
        d4 = torch.cat([d4, e3_att], dim=1)

        d3 = self.up3(d4)                       # 16 → 32
        e2_att = self.att3(gate=d3, skip=e2)    # Attention on e2
        d3 = torch.cat([d3, e2_att], dim=1)

        d2 = self.up2(d3)                       # 32 → 64
        e1_att = self.att2(gate=d2, skip=e1)    # Attention on e1
        d2 = torch.cat([d2, e1_att], dim=1)

        d1 = self.up1(d2)                       # 64 → 128
        e0_att = self.att1(gate=d1, skip=e0)    # Attention on e0
        d1 = torch.cat([d1, e0_att], dim=1)

        out = self.final(d1)                    # 128×128 output

        # --------- FINAL UPSAMPLING TO 256×256 ---------
        out = F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=False)
        
        return torch.sigmoid(out)


# Import loss functions from efficientunet_model
from efficientunet_model import (
    DiceBCELoss, DiceFocalLoss, DiceTverskyLoss, 
    dice_coefficient, iou_score, pixel_accuracy
)
