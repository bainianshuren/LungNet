import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.lungnet import LungNet
from data.dataset import get_dataloader
from utils.logger import setup_logger
from utils.metrics import calculate_map
from utils.common import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description='LungNet training script')
    parser.add_argument('--config', type=str, default='configs/lungnet.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--dataset', type=str, help='Override dataset name')
    parser.add_argument('--data_path', type=str, help='Override data path')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--device', type=str, default=None, help='Device to use')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    args = parse_args()
    config = load_config(args.config)

    # Override config with command-line arguments if provided
    if args.dataset:
        config['data']['dataset'] = args.dataset
    if args.data_path:
        config['data']['root'] = args.data_path
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.epochs:
        config['train']['epochs'] = args.epochs
    if args.lr:
        config['train']['lr'] = args.lr

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Set random seed
    set_seed(config['train']['seed'])

    # Setup logger
    logger = setup_logger('train', 'logs/train_logs.txt')
    logger.info(f'Configuration: {config}')

    # Initialize model
    model = LungNet(pretrained=True).to(device)
    model.train()

    # Data loaders
    train_loader = get_dataloader(
        config['data']['root'],
        config['data']['dataset'],
        'train',
        config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    val_loader = get_dataloader(
        config['data']['root'],
        config['data']['dataset'],
        'test',
        config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )

    # Optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['train']['lr_step'],
        gamma=config['train']['lr_gamma']
    )

    # Training loop
    best_map = 0.0
    early_stop_count = 0
    epochs = config['train']['epochs']

    for epoch in range(epochs):
        logger.info(f'Epoch [{epoch+1}/{epochs}]')
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        for imgs, boxes in pbar:
            imgs = imgs.to(device)
            boxes = [b.to(device) for b in boxes]

            optimizer.zero_grad()
            loss = model.train_step((imgs, boxes), device)['loss']
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': train_loss / len(pbar)})

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_map = calculate_map(model, val_loader, device)
        logger.info(f'Validation mAP@0.5: {val_map:.4f}')

        # Save best model
        if val_map > best_map:
            best_map = val_map
            os.makedirs('weights', exist_ok=True)
            torch.save(model.state_dict(), 'weights/best_LungNet.pth')
            logger.info(f'Saved best model with mAP: {best_map:.4f}')
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= config['train']['early_stop_patience']:
                logger.info(f'Early stopping at epoch {epoch+1}')
                break

    logger.info(f'Training completed. Best mAP: {best_map:.4f}')

if __name__ == '__main__':
    main()