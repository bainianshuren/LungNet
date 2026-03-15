import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from torchvision import transforms

class LungNoduleDataset(Dataset):
    """Lung Nodule Detection Dataset"""
    def __init__(self, data_path, dataset_type='LUNA16', split='train', augment=True):
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.split = split
        self.augment = augment

        # Load image and label paths
        self.img_paths = [os.path.join(data_path, 'images', split, f) for f in os.listdir(os.path.join(data_path, 'images', split)) if f.endswith('.png')]
        self.label_paths = [os.path.join(data_path, 'labels', split, f.replace('.png', '.txt')) for f in os.listdir(os.path.join(data_path, 'images', split)) if f.endswith('.png')]

        # Data Augmentation
        self.transform = self._get_transforms()

    def _get_transforms(self):
        """Obtain data augmentation strategies"""
        transform_list = []
        if self.augment and self.split == 'train':
            transform_list.extend([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ])
        else:
            transform_list.extend([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])
        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load Image
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=-1)  # [H, W, 1]

        # Load Tags
        label_path = self.label_paths[idx]
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    xmin, ymin, xmax, ymax, cls = map(float, line.strip().split())
                    boxes.append([xmin, ymin, xmax, ymax, cls])
        boxes = torch.tensor(boxes) if boxes else torch.zeros((0, 5))

        # Data Augmentation
        if self.transform:
            img = self.transform(img)

        return img, boxes

def get_dataloader(data_path, dataset_type='LUNA16', split='train', batch_size=8, num_workers=4):
    """Get data loader"""
    dataset = LungNoduleDataset(data_path, dataset_type, split)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if split == 'train' else False,
        num_workers=num_workers,
        collate_fn=lambda x: (torch.stack([i[0] for i in x]), [i[1] for i in x])
    )
    return dataloader
