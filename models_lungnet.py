import torch
import torch.nn as nn
from ultralytics import YOLO
from .mdcn import MDCN
from .dwa_conv import DWA_Conv
from .aaf_net import AAF_Net

class LungNet(nn.Module):
    """LungNet主模型：基于YOLOv11 + MDCN + DWA-Conv + AAF-Net"""
    def __init__(self, pretrained=True):
        super(LungNet, self).__init__()
        # 加载YOLOv11基线模型
        self.yolov11 = YOLO('yolov11n.pt') if pretrained else YOLO('yolov11n.yaml')
        self.backbone = self.yolov11.model.backbone
        self.neck = self.yolov11.model.neck
        self.head = self.yolov11.model.head

        # 替换Backbone中的卷积为MDCN
        self._replace_backbone_with_mdcn()

        # 替换Neck中的卷积为DWA-Conv
        self._replace_neck_with_dwa_conv()

        # 添加AAF-Net作为检测头前置模块
        self.aaf_net = AAF_Net(in_channels_list=[64, 128, 256], out_channels=256)

    def _replace_backbone_with_mdcn(self):
        """替换Backbone中的标准卷积为MDCN"""
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3):
                mdcn = MDCN(module.in_channels, module.out_channels, kernel_sizes=[3,5,7])
                setattr(self.backbone, name, mdcn)

    def _replace_neck_with_dwa_conv(self):
        """替换Neck中的标准卷积为DWA-Conv"""
        for name, module in self.neck.named_modules():
            if isinstance(module, nn.Conv2d) and module.dilation == 1:
                dwa_conv = DWA_Conv(module.in_channels, module.out_channels, rates=[1,2,4])
                setattr(self.neck, name, dwa_conv)

    def forward(self, x):
        # Backbone提取特征
        backbone_feats = self.backbone(x)
        
        # Neck特征融合（DWA-Conv）
        neck_feats = self.neck(backbone_feats)
        
        # AAF-Net解剖感知融合
        aaf_feats = self.aaf_net(neck_feats)
        
        # 检测头预测
        out = self.head(aaf_feats)
        return out

    def train_step(self, batch, device):
        """训练步（适配YOLOv11训练逻辑）"""
        return self.yolov11.model.train_step(batch, device)

    def val_step(self, batch, device):
        """验证步"""
        return self.yolov11.model.val_step(batch, device)
