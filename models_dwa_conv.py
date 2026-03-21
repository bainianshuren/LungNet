import torch
import torch.nn as nn
import torch.nn.functional as F

class DWA_Conv(nn.Module):
    """Dynamic Weight-Assigned Atrous Convolution（Dynamic Weight-Assigned Atrous Convolution）"""
    def __init__(self, in_channels, out_channels, rates=[1, 2, 4], stride=1):
        super(DWA_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rates = rates
        self.stride = stride

        # Dilated Convolution Branch
        self.atrous_branches = nn.ModuleList()
        for r in rates:
            self.atrous_branches.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=r, dilation=r, bias=False)
            )

        # Dynamic Weight Prediction (Based on Spatial and Channel Information of Features)
        self.weight_pred = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, len(rates), kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # Predictive Branch Weight
        weights = self.weight_pred(x)  # [B, N, H, W] N is the number of dilation rates

        # Dilated Convolution Calculation + Weighted Weighting
        out = 0
        for i, conv in enumerate(self.atrous_branches):
            branch_out = conv(x)  # [B, C, H, W]
            out += branch_out * weights[:, i:i+1, :, :]  # Spatial Dimension Weighting

        out = self.bn(out)
        out = self.act(out)
        return out
