import os
import argparse
import pydicom
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils.luna16_utils import convert_luna16_annotations

def parse_args():
    parser = argparse.ArgumentParser(description='Lung nodule dataset preprocessing')
    parser.add_argument('--dataset', type=str, required=True, choices=['LUNA16', 'Lung-PET-CT-Dx'],
                        help='Dataset name')
    parser.add_argument('--raw_path', type=str, required=True, help='Raw dataset path')
    parser.add_argument('--save_path', type=str, required=True, help='Preprocessed data save path')
    return parser.parse_args()

def load_dicom(path):
    """Load a DICOM file and convert to a normalized numpy array."""
    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array
    # Normalize to [0,1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def preprocess_luna16(raw_path, save_path):
    """Preprocess the LUNA16 dataset."""
    # Create output directories
    os.makedirs(os.path.join(save_path, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'images/test'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'labels/test'), exist_ok=True)

    # Assume annotations.csv is in raw_path
    csv_path = os.path.join(raw_path, 'annotations.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"annotations.csv not found in {raw_path}")

    df = pd.read_csv(csv_path)
    unique_series = df['seriesuid'].unique()
    train_series, test_series = train_test_split(unique_series, test_size=0.3, random_state=42)
    train_df = df[df['seriesuid'].isin(train_series)]
    test_df = df[df['seriesuid'].isin(test_series)]

    # Save temporary CSV for train and test
    train_csv = os.path.join(save_path, 'train_annotations.csv')
    test_csv = os.path.join(save_path, 'test_annotations.csv')
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Convert train and test
    convert_luna16_annotations(train_csv, raw_path,
                               os.path.join(save_path, 'images/train'),
                               os.path.join(save_path, 'labels/train'))
    convert_luna16_annotations(test_csv, raw_path,
                               os.path.join(save_path, 'images/test'),
                               os.path.join(save_path, 'labels/test'))

    print(f'LUNA16 preprocessing completed. Data saved to {save_path}')

def preprocess_lung_pet_ct(raw_path, save_path):
    """Preprocess the Lung-PET-CT-Dx dataset."""
    # Create output directories
    os.makedirs(os.path.join(save_path, 'ct/train'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'ct/test'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'pet/train'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'pet/test'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'labels/test'), exist_ok=True)

    # Assume CT and PET folders exist under raw_path
    ct_dir = os.path.join(raw_path, 'CT')
    pet_dir = os.path.join(raw_path, 'PET')
    if not os.path.isdir(ct_dir) or not os.path.isdir(pet_dir):
        raise FileNotFoundError(f"CT or PET folder not found in {raw_path}")

    ct_paths = [os.path.join(ct_dir, f) for f in os.listdir(ct_dir) if f.endswith('.dcm')]
    pet_paths = [os.path.join(pet_dir, f) for f in os.listdir(pet_dir) if f.endswith('.dcm')]

    # Split by patient (simplified: split file lists)
    train_ct, test_ct = train_test_split(ct_paths, test_size=0.3, random_state=42)
    train_pet, test_pet = train_test_split(pet_paths, test_size=0.3, random_state=42)

    # Process CT train
    for idx, path in tqdm(enumerate(train_ct), desc='Processing CT Train'):
        img = load_dicom(path)
        img = cv2.resize(img, (512, 512))
        cv2.imwrite(os.path.join(save_path, f'ct/train/{idx}.png'), (img * 255).astype(np.uint8))
    # Process CT test
    for idx, path in tqdm(enumerate(test_ct), desc='Processing CT Test'):
        img = load_dicom(path)
        img = cv2.resize(img, (512, 512))
        cv2.imwrite(os.path.join(save_path, f'ct/test/{idx}.png'), (img * 255).astype(np.uint8))

    # Process PET train
    for idx, path in tqdm(enumerate(train_pet), desc='Processing PET Train'):
        img = load_dicom(path)
        img = cv2.resize(img, (200, 200))
        cv2.imwrite(os.path.join(save_path, f'pet/train/{idx}.png'), (img * 255).astype(np.uint8))
    # Process PET test
    for idx, path in tqdm(enumerate(test_pet), desc='Processing PET Test'):
        img = load_dicom(path)
        img = cv2.resize(img, (200, 200))
        cv2.imwrite(os.path.join(save_path, f'pet/test/{idx}.png'), (img * 255).astype(np.uint8))

    # Note: Label generation for Lung-PET-CT-Dx requires annotation files.
    # Please adapt the code to convert your specific annotation format to YOLO format.
    print(f'Lung-PET-CT-Dx preprocessing completed. Data saved to {save_path}')

def main():
    args = parse_args()
    if args.dataset == 'LUNA16':
        preprocess_luna16(args.raw_path, args.save_path)
    elif args.dataset == 'Lung-PET-CT-Dx':
        preprocess_lung_pet_ct(args.raw_path, args.save_path)
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')

if __name__ == '__main__':
    main()