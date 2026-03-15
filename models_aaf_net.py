import torch
import torch.nn as nn
import torch.nn.functional as F

class AAF_Net(nn.Module):
    """Anatomy-Aware Dual-Path Fusion Network（Anatomy-Aware Dual-Path Fusion Network）"""
    def __init__(self, in_channels_list, out_channels):
        super(AAF_Net, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        # Path 1: Anatomical Context Extraction (Large Receptive Field)
        self.anatomy_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(None),
            nn.Conv2d(sum(in_channels_list), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

        # Path 2: Detail Feature Preservation (Small Receptive Field)
        self.detail_path = nn.ModuleList()
        for ch in in_channels_list:
            self.detail_path.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU()
                )
            )

        # fusion weight
        self.fusion_weight = nn.Conv2d(out_channels * 2, 2, kernel_size=1)
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x_list):
        # Feature Alignment
        x_aligned = []
        target_size = x_list[0].shape[2:]
        for x in x_list:
            x_aligned.append(F.interpolate(x, size=target_size, mode='bilinear', align_corners=False))
        
        # Path 1: Anatomical Context
        x_concat = torch.cat(x_aligned, dim=1)
        anatomy_feat = self.anatomy_path(x_concat)

        # Path 2: Detail Features
        detail_feat = 0
        for i, x in enumerate(x_aligned):
            detail_feat += self.detail_path[i](x)
        detail_feat /= len(x_aligned)

        # Dynamic Fusion
        fusion_input = torch.cat([anatomy_feat, detail_feat], dim=1)
        weights = F.softmax(self.fusion_weight(fusion_input), dim=1)
        fused_feat = anatomy_feat * weights[:, 0:1, :, :] + detail_feat * weights[:, 1:2, :, :]

        # Final Integration
        out = self.final_conv(fused_feat)
        out = self.bn(out)
        out = self.act(out)
        return out
