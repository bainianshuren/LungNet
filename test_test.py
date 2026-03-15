import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from models.lungnet import LungNet
from data.dataset import get_dataloader
from utils.metrics import calculate_map, calculate_recall, get_model_flops, get_model_params
from utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description='LungNet testing script')
    parser.add_argument('--dataset', type=str, required=True, choices=['LUNA16', 'Lung-PET-CT-Dx'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--weight_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def calculate_fps(model, test_loader, device, warmup=10):
    """Calculate FPS"""
    model.eval()
    # Warmup
    for i, (imgs, _) in enumerate(test_loader):
        if i >= warmup:
            break
        imgs = imgs.to(device)
        with torch.no_grad():
            model(imgs)
    
    # timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_imgs = 0
    start.record()
    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc='Calculating FPS'):
            imgs = imgs.to(device)
            model(imgs)
            total_imgs += imgs.shape[0]
    end.record()
    torch.cuda.synchronize()
    time = start.elapsed_time(end) / 1000  # 秒
    fps = total_imgs / time
    return fps

def main():
    args = parse_args()
    logger = setup_logger('test', 'test_logs.txt')

    # Load model
    model = LungNet(pretrained=False).to(args.device)
    model.load_state_dict(torch.load(args.weight_path, map_location=args.device))
    model.eval()

    # Load data
    test_loader = get_dataloader(args.data_path, args.dataset, 'test', batch_size=8)

    # Calculate evaluation metrics
    logger.info('Calculating evaluation metrics...')
    map_05 = calculate_map(model, test_loader, args.device, iou_thres=0.5)
    map_05_95 = calculate_map(model, test_loader, args.device, iou_thres=np.arange(0.5, 1.0, 0.05))
    recall = calculate_recall(model, test_loader, args.device)
    fps = calculate_fps(model, test_loader, args.device)
    params = get_model_params(model)
    flops = get_model_flops(model, input_size=(1, 1, 330, 330))

    # Output Result
    logger.info(f'===== Test Results =====')
    logger.info(f'mAP@0.5: {map_05:.4f}')
    logger.info(f'mAP@0.5:0.95: {map_05_95:.4f}')
    logger.info(f'mRecall: {recall:.4f}')
    logger.info(f'FPS: {fps:.2f} img/s')
    logger.info(f'Params: {params:.2f} M')
    logger.info(f'GFLOPs: {flops:.2f}')

if __name__ == '__main__':
    main()
