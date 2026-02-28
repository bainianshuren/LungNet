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

    # 1. 初始化模型
    model = LungNet(pretrained=True).to(args.device)
    model.train()

    # 2. 数据加载
    train_loader = get_dataloader(args.data_path, args.dataset, 'train', args.batch_size)
    val_loader = get_dataloader(args.data_path, args.dataset, 'test', args.batch_size)

    # 3. 优化器与损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)  # 15epoch后lr降为1e-5

    # 4. 训练循环
    best_map = 0.0
    early_stop_count = 0
    for epoch in range(args.epochs):
        logger.info(f'Epoch [{epoch+1}/{args.epochs}]')
        train_loss = 0.0

        # 训练步
        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        for imgs, boxes in pbar:
            imgs = imgs.to(args.device)
            boxes = [b.to(args.device) for b in boxes]

            optimizer.zero_grad()
            outputs = model(imgs)
            # 计算损失（适配YOLOv11损失逻辑）
            loss = model.train_step((imgs, boxes), args.device)['loss']
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': train_loss / (len(pbar))})

        # 学习率调度
        scheduler.step()

        # 验证步
        model.eval()
        with torch.no_grad():
            val_map = calculate_map(model, val_loader, args.device)
        logger.info(f'Validation mAP@0.5: {val_map:.4f}')

        # 保存最优权重
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
