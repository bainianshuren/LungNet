import torch
import torch.nn as nn
import torch.nn.functional as F

class MDCN(nn.Module):
    """（Multi-dimensional Dynamic Convolution Module）"""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], stride=1, padding='same'):
        super(MDCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.stride = stride
        self.padding = padding

        # Dynamic Weight Generator (Adaptive to Weights of Different Convolutional Kernels)
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, len(kernel_sizes), kernel_size=1),
            nn.Softmax(dim=1)
        )

        # Multi-scale Convolutional Branch
        self.conv_branches = nn.ModuleList()
        for k in kernel_sizes:
            pad = k // 2 if padding == 'same' else 0
            self.conv_branches.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=stride, padding=pad, bias=False)
            )

        # Channel Fusion
        self.channel_fusion = nn.Conv2d(out_channels * len(kernel_sizes), out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # Generate dynamic weights
        weights = self.weight_generator(x)  # [B, N, 1, 1] N for the number of convolution kernels
        weights = weights.unsqueeze(2).unsqueeze(3)  # [B, N, 1, 1, 1]

        # Multi-scale Convolution Calculation
        branch_outs = []
        for i, conv in enumerate(self.conv_branches):
            out = conv(x)  # [B, C, H, W]
            branch_outs.append(out * weights[:, i])  # weighted

        # Integration
        concat_out = torch.cat(branch_outs, dim=1)
        out = self.channel_fusion(concat_out)
        out = self.bn(out)
        out = self.act(out)
        return out
