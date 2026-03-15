import torch
import torch.nn as nn
from ultralytics import YOLO
from .mdcn import MDCN
from .dwa_conv import DWA_Conv
from .aaf_net import AAF_Net

class LungNet(nn.Module):
    """LungNet：YOLOv11 + MDCN + DWA-Conv + AAF-Net"""
    def __init__(self, pretrained=True):
        super(LungNet, self).__init__()
        # Load YOLOv11 baseline model
        self.yolov11 = YOLO('yolov11n.pt') if pretrained else YOLO('yolov11n.yaml')
        self.backbone = self.yolov11.model.backbone
        self.neck = self.yolov11.model.neck
        self.head = self.yolov11.model.head

        # Replace the convolution in the Backbone with MDCN
        self._replace_backbone_with_mdcn()

        # Replace the convolution in Neck with DWA-Conv
        self._replace_neck_with_dwa_conv()

        # Add AAF-Net as a pre-module of the detection head
        self.aaf_net = AAF_Net(in_channels_list=[64, 128, 256], out_channels=256)

    def _replace_backbone_with_mdcn(self):
        """Replace the standard convolution in the Backbone with MDCN"""
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3):
                mdcn = MDCN(module.in_channels, module.out_channels, kernel_sizes=[3,5,7])
                setattr(self.backbone, name, mdcn)

    def _replace_neck_with_dwa_conv(self):
        """Replace the standard convolution in Neck with DWA-Conv"""
        for name, module in self.neck.named_modules():
            if isinstance(module, nn.Conv2d) and module.dilation == 1:
                dwa_conv = DWA_Conv(module.in_channels, module.out_channels, rates=[1,2,4])
                setattr(self.neck, name, dwa_conv)

    def forward(self, x):
        # Backbone extract features
        backbone_feats = self.backbone(x)
        
        # Neck feature fusion（DWA-Conv）
        neck_feats = self.neck(backbone_feats)
        
        # AAF-Net Anatomical Perception Fusion
        aaf_feats = self.aaf_net(neck_feats)
        
        # Detection head prediction
        out = self.head(aaf_feats)
        return out

    def train_step(self, batch, device):
        return self.yolov11.model.train_step(batch, device)

    def val_step(self, batch, device):
        return self.yolov11.model.val_step(batch, device)
