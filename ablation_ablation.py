import os
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
from models.lungnet import LungNet
from models.mdcn import MDCN
from models.dwa_conv import DWA_Conv
from models.aaf_net import AAF_Net
from data.dataset import get_dataloader
from utils.metrics import calculate_map
from utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description='LungNet ablation experiment')
    parser.add_argument('--dataset', type=str, required=True, choices=['LUNA16', 'Lung-PET-CT-Dx'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--modules', type=str, nargs='+', choices=['MDCN', 'DWA-Conv', 'AAF-Net'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def build_ablation_model(modules, device):
    """Construct ablation experiment model"""
    # Basic YOLOv11 Model
    model = LungNet(pretrained=True).to(device)
    
    # Replace/Add according to the specified module
    if 'MDCN' not in modules:
        # Remove MDCN and restore the original convolution
        for name, module in model.backbone.named_modules():
            if isinstance(module, MDCN):
                orig_conv = torch.nn.Conv2d(module.in_channels, module.out_channels, kernel_size=3, padding=1)
                setattr(model.backbone, name, orig_conv)
    
    if 'DWA-Conv' not in modules:
        # Remove DWA-Conv and restore the original convolution
        for name, module in model.neck.named_modules():
            if isinstance(module, DWA_Conv):
                orig_conv = torch.nn.Conv2d(module.in_channels, module.out_channels, kernel_size=3, padding=1)
                setattr(model.neck, name, orig_conv)
    
    if 'AAF-Net' not in modules:
        model.aaf_net = torch.nn.Identity()
    
    return model

def main():
    args = parse_args()
    logger = setup_logger('ablation', 'ablation_logs.txt')
    logger.info(f'Ablation experiment with modules: {args.modules}')

    # Construct ablation models
    model = build_ablation_model(args.modules, args.device)
    model.train()

    # Data loading
    train_loader = get_dataloader(args.data_path, args.dataset, 'train', args.batch_size)
    val_loader = get_dataloader(args.data_path, args.dataset, 'test', args.batch_size)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # training
    best_map = 0.0
    for epoch in range(args.epochs):
        logger.info(f'Epoch [{epoch+1}/{args.epochs}]')
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        for imgs, boxes in pbar:
            imgs = imgs.to(args.device)
            boxes = [b.to(args.device) for b in boxes]

            optimizer.zero_grad()
            loss = model.train_step((imgs, boxes), args.device)['loss']
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': train_loss / len(pbar)})

        scheduler.step()

        # Verification
        model.eval()
        with torch.no_grad():
            val_map = calculate_map(model, val_loader, args.device)
        logger.info(f'Validation mAP@0.5: {val_map:.4f}')

        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), f'weights/ablation_{"_".join(args.modules)}.pth')

    logger.info(f'Ablation experiment completed! Best mAP: {best_map:.4f}')

if __name__ == '__main__':
    main()
