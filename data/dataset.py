import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import re


class SodaCanDataset(Dataset):
    """Dataset для банок газировки с поддержкой grayscale"""

    def __init__(self, data_dir, labels_file, phase='train', transform=None):
        self.data_dir = Path(data_dir)
        self.phase = phase
        self.transform = transform
        self.samples = self._load_labels(labels_file)

        # Импортируем здесь чтобы избежать циклических импортов
        from data.transforms import TiltAugmentation
        self.tilt_augmentation = TiltAugmentation() if phase == 'train' else None

        print(f"Loaded {len(self.samples)} samples for {phase}")

    def _load_labels(self, labels_file):
        """Загружает разметку из файла"""
        samples = []

        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Парсим строку: "snap_2025-11-17_14-54-18 : 0"
                match = re.match(r'(.+?)\s*:\s*(\d+)', line)
                if match:
                    filename = match.group(1)
                    angle = int(match.group(2))

                    # ИСПРАВЛЕНИЕ: заменяем 360 на 0 (т.к. у нас классы 0-359)
                    if angle == 360:
                        angle = 0
                    # Также проверяем, что угол в допустимом диапазоне
                    if angle < 0 or angle >= 360:
                        print(f"Warning: Invalid angle {angle} in {filename}, skipping")
                        continue

                    # Проверяем существование файла (поддерживаем разные форматы)
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        img_path = self.data_dir / f"{filename}{ext}"
                        if img_path.exists():
                            samples.append((img_path, angle))
                            break
                    else:
                        print(f"Warning: {filename} not found with common extensions")

        # Дополнительная проверка: выводим информацию об углах
        angles = [angle for _, angle in samples]
        print(f"Angle range: {min(angles)}-{max(angles)}")
        print(f"Unique angles: {len(set(angles))}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from config.config import cfg

        img_path, angle = self.samples[idx]

        # Загружаем изображение
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)  # Сразу загружаем как grayscale

        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {img_path}")

        # Добавляем канальное измерение (H, W) -> (H, W, 1)
        image = np.expand_dims(image, axis=-1)

        # Применяем специальную аугментацию наклона
        if self.phase == 'train' and self.tilt_augmentation:
            image = self.tilt_augmentation.apply_tilt(image, cfg.MAX_TILT_ANGLE)

        # Применяем стандартные трансформы
        if self.transform:
            image = self.transform(image=image)['image']

        # Преобразуем угол в циклические координаты
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