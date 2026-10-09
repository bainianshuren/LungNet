# utils/luna16_utils.py
# Utility functions for converting LUNA16 annotations.csv to YOLO format.

import os
import numpy as np
import pandas as pd
import pydicom
import cv2
from scipy.ndimage import zoom

def load_dicom_series(series_dir):
    """Load a DICOM series and return a 3D array with metadata."""
    slices = [pydicom.dcmread(os.path.join(series_dir, f)) for f in os.listdir(series_dir) if f.endswith('.dcm')]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    volume = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
    return volume, slices

def world_to_pixel(coord_x, coord_y, coord_z, origin, spacing):
    """Convert world coordinates to pixel coordinates."""
    x = (coord_x - origin[0]) / spacing[0]
    y = (coord_y - origin[1]) / spacing[1]
    z = (coord_z - origin[2]) / spacing[2]
    return int(round(x)), int(round(y)), int(round(z))

def get_max_slice_and_bbox(volume, nodule_info, origin, spacing):
    """
    Find the slice with the maximum cross-section and generate the bounding box.
    nodule_info: dict with keys 'coordX', 'coordY', 'coordZ', 'diameter_mm'
    """
    x, y, z = world_to_pixel(nodule_info['coordX'], nodule_info['coordY'], nodule_info['coordZ'], origin, spacing)
    diameter_px = nodule_info['diameter_mm'] / spacing[0]
    half = diameter_px / 2.0
    xmin = max(0, x - half)
    ymin = max(0, y - half)
    xmax = min(volume.shape[2], x + half)
    ymax = min(volume.shape[1], y + half)
    slice_idx = int(round(z))
    return slice_idx, (xmin, ymin, xmax, ymax)

def convert_luna16_annotations(csv_path, raw_data_root, save_img_dir, save_label_dir):
    """
    Convert LUNA16 annotations.csv to YOLO format.
    CSV columns: seriesuid, coordX, coordY, coordZ, diameter_mm
    """
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_label_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    for seriesuid, group in df.groupby('seriesuid'):
        series_dir = os.path.join(raw_data_root, seriesuid)
        if not os.path.isdir(series_dir):
            continue
        volume, slices = load_dicom_series(series_dir)
        origin = slices[0].ImagePositionPatient
        spacing = slices[0].PixelSpacing + [slices[0].SliceThickness]
        for idx, row in group.iterrows():
            slice_idx, bbox = get_max_slice_and_bbox(volume, row, origin, spacing)
            img = volume[slice_idx]
            # Normalize to [0,1] and resize to 330x330
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = zoom(img, (330 / img.shape[0], 330 / img.shape[1]), order=1)
            # Scale bbox coordinates
            scale_x = 330 / volume.shape[2]
            scale_y = 330 / volume.shape[1]
            xmin, ymin, xmax, ymax = bbox
            xmin, xmax = xmin * scale_x, xmax * scale_x
            ymin, ymax = ymin * scale_y, ymax * scale_y
            # Save image
            img_name = f"{seriesuid}_{slice_idx}.png"
            cv2.imwrite(os.path.join(save_img_dir, img_name), (img * 255).astype(np.uint8))
            # Save YOLO label: class x_center y_center width height (normalized)
            x_center = (xmin + xmax) / 2.0 / 330
            y_center = (ymin + ymax) / 2.0 / 330
            w = (xmax - xmin) / 330
            h = (ymax - ymin) / 330
            with open(os.path.join(save_label_dir, img_name.replace('.png', '.txt')), 'w') as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")