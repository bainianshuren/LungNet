import os
import argparse
import torch
import cv2
import numpy as np
from models.lungnet import LungNet
from utils.visualization import draw_bbox

def parse_args():
    parser = argparse.ArgumentParser(description='LungNet inference script')
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--weight_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def load_image(img_path):
    """Load and preprocess images"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (330, 330))
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return img, cv2.imread(img_path)

def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    # Load model
    model = LungNet(pretrained=False).to(args.device)
    model.load_state_dict(torch.load(args.weight_path, map_location=args.device))
    model.eval()

    # Process single/batch images
    if os.path.isdir(args.img_path):
        img_paths = [os.path.join(args.img_path, f) for f in os.listdir(args.img_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    else:
        img_paths = [args.img_path]

    for img_path in img_paths:
        # Load Image
        img_tensor, img_vis = load_image(img_path)
        img_tensor = img_tensor.to(args.device)

        # inference
        with torch.no_grad():
            outputs = model(img_tensor)
            # Parse output (compatible with YOLOv11 output format)
            boxes = outputs[0].cpu().numpy()  # [xmin, ymin, xmax, ymax, conf, cls]

        # Visualization
        for box in boxes:
            if box[4] > 0.5:  # confidence threshold
                xmin, ymin, xmax, ymax, conf, cls = box
                img_vis = draw_bbox(img_vis, int(xmin), int(ymin), int(xmax), int(ymax), f'Nodule {conf:.2f}')

        # Save results
        save_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(args.save_path, save_name), img_vis)
        print(f'Inference result saved to {os.path.join(args.save_path, save_name)}')

if __name__ == '__main__':
    main()
