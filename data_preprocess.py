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
    """Load DICOM files and convert them to numpy arrays"""
    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array
    # Normalize to [0,1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def preprocess_luna16(raw_path, save_path):
    """Preprocess the LUNA16 dataset"""
    # Create a save directory
    os.makedirs(os.path.join(save_path, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'images/test'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'labels/test'), exist_ok=True)

    # Traverse the original data (simplified logic, need to adjust according to the actual LUNA16 structure)
    dicom_paths = [os.path.join(raw_path, f) for f in os.listdir(raw_path) if f.endswith('.dcm')]
    train_paths, test_paths = train_test_split(dicom_paths, test_size=0.3, random_state=42)

    # Process the training set
    for idx, path in tqdm(enumerate(train_paths), desc='Processing LUNA16 Train'):
        img = load_dicom(path)
        img = cv2.resize(img, (330, 330))
        img_save_path = os.path.join(save_path, f'images/train/{idx}.png')
        cv2.imwrite(img_save_path, img * 255)
        # Tag processing
        label_save_path = os.path.join(save_path, f'labels/train/{idx}.txt')
        # annotation format：xmin ymin xmax ymax class
        # example：0 0 50 50 0
        with open(label_save_path, 'w') as f:
            pass  # Replace with actual annotation logic

    # Process the test set (using the same logic as the training set)
    for idx, path in tqdm(enumerate(test_paths), desc='Processing LUNA16 Test'):
        img = load_dicom(path)
        img = cv2.resize(img, (330, 330))
        img_save_path = os.path.join(save_path, f'images/test/{idx}.png')
        cv2.imwrite(img_save_path, img * 255)
        label_save_path = os.path.join(save_path, f'labels/test/{idx}.txt')
        with open(label_save_path, 'w') as f:
            pass

def preprocess_lung_pet_ct(raw_path, save_path):
    """preprocessing Lung-PET-CT-Dx"""
    # Create a save directory
    os.makedirs(os.path.join(save_path, 'ct/train'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'ct/test'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'pet/train'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'pet/test'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'labels/test'), exist_ok=True)

    # Traverse the original data (simplified logic)
    ct_paths = [os.path.join(raw_path, 'CT', f) for f in os.listdir(os.path.join(raw_path, 'CT')) if f.endswith('.dcm')]
    pet_paths = [os.path.join(raw_path, 'PET', f) for f in os.listdir(os.path.join(raw_path, 'PET')) if f.endswith('.dcm')]
    train_ct, test_ct = train_test_split(ct_paths, test_size=0.3, random_state=42)
    train_pet, test_pet = train_test_split(pet_paths, test_size=0.3, random_state=42)

    # Process CT training set (512x512)
    for idx, path in tqdm(enumerate(train_ct), desc='Processing PET-CT CT Train'):
        img = load_dicom(path)
        img = cv2.resize(img, (512, 512))
        cv2.imwrite(os.path.join(save_path, f'ct/train/{idx}.png'), img * 255)

    # Process PET training set (200x200)
    for idx, path in tqdm(enumerate(train_pet), desc='Processing PET-CT PET Train'):
        img = load_dicom(path)
        img = cv2.resize(img, (200, 200))
        cv2.imwrite(os.path.join(save_path, f'pet/train/{idx}.png'), img * 255)

    # Process the test set (same as the training set)
    for idx, path in enumerate(test_ct):
        img = load_dicom(path)
        img = cv2.resize(img, (512, 512))
        cv2.imwrite(os.path.join(save_path, f'ct/test/{idx}.png'), img * 255)
    for idx, path in enumerate(test_pet):
        img = load_dicom(path)
        img = cv2.resize(img, (200, 200))
        cv2.imwrite(os.path.join(save_path, f'pet/test/{idx}.png'), img * 255)

    # Tag matching (needs to be implemented based on the actual annotation file)
def main():
    args = parse_args()
    if args.dataset == 'LUNA16':
        preprocess_luna16(args.raw_path, args.save_path)
    elif args.dataset == 'Lung-PET-CT-Dx':
        preprocess_lung_pet_ct(args.raw_path, args.save_path)
    print(f'Preprocessing {args.dataset} dataset completed! Save to {args.save_path}')

if __name__ == '__main__':
    main()
