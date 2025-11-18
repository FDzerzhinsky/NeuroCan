import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import re

from config import cfg
from data import get_transforms, TiltAugmentation


class SodaCanDataset(Dataset):
    """Dataset для банок газировки"""

    def __init__(self, data_dir, labels_file, phase='train', transform=None):
        self.data_dir = Path(data_dir)
        self.phase = phase
        self.transform = transform
        self.samples = self._load_labels(labels_file)
        self.tilt_augmentation = TiltAugmentation() if phase == 'train' else None

        print(f"Loaded {len(self.samples)} samples for {phase}")

    def _load_labels(self, labels_file):
        samples = []

        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                match = re.match(r'(.+?)\s*:\s*(\d+)', line)
                if match:
                    filename = match.group(1)
                    angle = int(match.group(2))

                    img_path = self.data_dir / f"{filename}.jpg"
                    if img_path.exists():
                        samples.append((img_path, angle))
                    else:
                        print(f"Warning: {img_path} not found")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, angle = self.samples[idx]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.phase == 'train' and self.tilt_augmentation:
            image = self.tilt_augmentation.apply_tilt(image, cfg.MAX_TILT_ANGLE)

        if self.transform:
            image = self.transform(image=image)['image']

        angle_rad = torch.tensor(angle) * 2 * torch.pi / 360
        sin_target = torch.sin(angle_rad)
        cos_target = torch.cos(angle_rad)

        return {
            'image': image,
            'angle': angle,
            'sin_target': sin_target,
            'cos_target': cos_target,
            'file_path': str(img_path)
        }