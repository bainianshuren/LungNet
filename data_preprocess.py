import os
import argparse
import pydicom
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description='Lung nodule dataset preprocessing')
    parser.add_argument('--dataset', type=str, required=True, choices=['LUNA16', 'Lung-PET-CT-Dx'],
                        help='Dataset name')
    parser.add_argument('--raw_path', type=str, required=True, help='Raw dataset path')
    parser.add_argument('--save_path', type=str, required=True, help='Preprocessed data save path')
    return parser.parse_args()

def load_dicom(path):
    """加载DICOM文件并转换为numpy数组"""
    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array
    # 归一化到[0,1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def preprocess_luna16(raw_path, save_path):
    """预处理LUNA16数据集"""
    # 创建保存目录
    os.makedirs(os.path.join(save_path, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'images/test'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'labels/test'), exist_ok=True)

    # 遍历原始数据（简化逻辑，需根据实际LUNA16结构调整）
    dicom_paths = [os.path.join(raw_path, f) for f in os.listdir(raw_path) if f.endswith('.dcm')]
    train_paths, test_paths = train_test_split(dicom_paths, test_size=0.3, random_state=42)

    # 处理训练集
    for idx, path in tqdm(enumerate(train_paths), desc='Processing LUNA16 Train'):
        img = load_dicom(path)
        # 裁剪到330x330
        img = cv2.resize(img, (330, 330))
        # 保存图像
        img_save_path = os.path.join(save_path, f'images/train/{idx}.png')
        cv2.imwrite(img_save_path, img * 255)
        # 标签处理（需根据实际标注文件生成，此处为占位）
        label_save_path = os.path.join(save_path, f'labels/train/{idx}.txt')
        # 标注格式：xmin ymin xmax ymax class
        # 示例：0 0 50 50 0
        with open(label_save_path, 'w') as f:
            pass  # 替换为实际标注逻辑

    # 处理测试集（同训练集逻辑）
    for idx, path in tqdm(enumerate(test_paths), desc='Processing LUNA16 Test'):
        img = load_dicom(path)
        img = cv2.resize(img, (330, 330))
        img_save_path = os.path.join(save_path, f'images/test/{idx}.png')
        cv2.imwrite(img_save_path, img * 255)
        label_save_path = os.path.join(save_path, f'labels/test/{idx}.txt')
        with open(label_save_path, 'w') as f:
            pass

def preprocess_lung_pet_ct(raw_path, save_path):
    """预处理Lung-PET-CT-Dx数据集"""
    # 创建保存目录
    os.makedirs(os.path.join(save_path, 'ct/train'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'ct/test'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'pet/train'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'pet/test'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'labels/test'), exist_ok=True)

    # 遍历原始数据（简化逻辑）
    ct_paths = [os.path.join(raw_path, 'CT', f) for f in os.listdir(os.path.join(raw_path, 'CT')) if f.endswith('.dcm')]
    pet_paths = [os.path.join(raw_path, 'PET', f) for f in os.listdir(os.path.join(raw_path, 'PET')) if f.endswith('.dcm')]
    train_ct, test_ct = train_test_split(ct_paths, test_size=0.3, random_state=42)
    train_pet, test_pet = train_test_split(pet_paths, test_size=0.3, random_state=42)

    # 处理CT训练集（512x512）
    for idx, path in tqdm(enumerate(train_ct), desc='Processing PET-CT CT Train'):
        img = load_dicom(path)
        img = cv2.resize(img, (512, 512))
        cv2.imwrite(os.path.join(save_path, f'ct/train/{idx}.png'), img * 255)

    # 处理PET训练集（200x200）
    for idx, path in tqdm(enumerate(train_pet), desc='Processing PET-CT PET Train'):
        img = load_dicom(path)
        img = cv2.resize(img, (200, 200))
        cv2.imwrite(os.path.join(save_path, f'pet/train/{idx}.png'), img * 255)

    # 处理测试集（同训练集）
    for idx, path in enumerate(test_ct):
        img = load_dicom(path)
        img = cv2.resize(img, (512, 512))
        cv2.imwrite(os.path.join(save_path, f'ct/test/{idx}.png'), img * 255)
    for idx, path in enumerate(test_pet):
        img = load_dicom(path)
        img = cv2.resize(img, (200, 200))
        cv2.imwrite(os.path.join(save_path, f'pet/test/{idx}.png'), img * 255)

    # 标签匹配（需根据实际标注文件实现）
    # ...

def main():
    args = parse_args()
    if args.dataset == 'LUNA16':
        preprocess_luna16(args.raw_path, args.save_path)
    elif args.dataset == 'Lung-PET-CT-Dx':
        preprocess_lung_pet_ct(args.raw_path, args.save_path)
    print(f'Preprocessing {args.dataset} dataset completed! Save to {args.save_path}')

if __name__ == '__main__':
    main()
