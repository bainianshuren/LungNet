import torch
import torch.nn as nn
import torch.nn.functional as F

class DWA_Conv(nn.Module):
    """动态权重分配空洞卷积（Dynamic Weight-Assigned Atrous Convolution）"""
    def __init__(self, in_channels, out_channels, rates=[1, 2, 4], stride=1):
        super(DWA_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rates = rates
        self.stride = stride

        # 空洞卷积分支
        self.atrous_branches = nn.ModuleList()
        for r in rates:
            self.atrous_branches.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=r, dilation=r, bias=False)
            )

        # 动态权重预测（基于特征的空间和通道信息）
        self.weight_pred = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, len(rates), kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # 预测分支权重
        weights = self.weight_pred(x)  # [B, N, H, W] N为空洞率数量

        # 空洞卷积计算 + 权重加权
        out = 0
        for i, conv in enumerate(self.atrous_branches):
            branch_out = conv(x)  # [B, C, H, W]
            out += branch_out * weights[:, i:i+1, :, :]  # 空间维度加权

        out = self.bn(out)
        out = self.act(out)
        return out
