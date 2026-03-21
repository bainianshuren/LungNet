import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.lungnet import LungNet
from data.dataset import get_dataloader
from utils.logger import setup_logger
from utils.metrics import calculate_map

def parse_args():
    parser = argparse.ArgumentParser(description='LungNet training script')
    parser.add_argument('--dataset', type=str, required=True, choices=['LUNA16', 'Lung-PET-CT-Dx'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger('train', 'train_logs.txt')

    # 1. Initialize model
    model = LungNet(pretrained=True).to(args.device)
    model.train()

    # 2. Data Loading
    train_loader = get_dataloader(args.data_path, args.dataset, 'train', args.batch_size)
    val_loader = get_dataloader(args.data_path, args.dataset, 'test', args.batch_size)

    # 3. Optimizer and Loss Function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)  # After 15 epochs, the learning rate (lr) is reduced to 1e-5

    # 4. training loop
    best_map = 0.0
    early_stop_count = 0
    for epoch in range(args.epochs):
        logger.info(f'Epoch [{epoch+1}/{args.epochs}]')
        train_loss = 0.0

    
        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        for imgs, boxes in pbar:
            imgs = imgs.to(args.device)
            boxes = [b.to(args.device) for b in boxes]

            optimizer.zero_grad()
            outputs = model(imgs)
            # Calculate Loss (Adapt to YOLOv11 Loss Logic)
            loss = model.train_step((imgs, boxes), args.device)['loss']
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': train_loss / (len(pbar))})

        # learning rate scheduling
        scheduler.step()

        # Verification
        model.eval()
        with torch.no_grad():
            val_map = calculate_map(model, val_loader, args.device)
        logger.info(f'Validation mAP@0.5: {val_map:.4f}')

        # Save Optimal Weights
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), 'weights/best_LungNet.pth')
            logger.info(f'Save best model with mAP: {best_map:.4f}')
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= 5:
                logger.info(f'Early stopping at epoch {epoch+1}')
                break

        model.train()

    logger.info(f'Training completed! Best mAP: {best_map:.4f}')

if __name__ == '__main__':
    main()
