import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2


def get_transforms(phase='train'):
    """
    Возвращает трансформы для обучения/валидации
    Теперь поддерживает grayscale изображения
    """
    from config.config import cfg  # Импортируем здесь чтобы избежать циклических импортов

    # Базовые трансформы которые применяются всегда
    base_transforms = []

    # Добавляем преобразование в grayscale если нужно
    if cfg.GRAYSCALE:
        base_transforms.append(A.ToGray(p=1.0))

    if phase == 'train':
        # Аугментации для обучения
        augmentations = [
            A.Affine(
                rotate=(-cfg.MAX_TILT_ANGLE, cfg.MAX_TILT_ANGLE),
                translate_percent=(-0.02, 0.02),
                scale=(0.98, 1.02),
                shear=(-1, 1),
                p=0.8
            ),
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=0.5
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.MotionBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
            ], p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
        ]

        # Нормализация для grayscale или RGB
        normalize = A.Normalize(
            mean=[0.485] if cfg.GRAYSCALE else [0.485, 0.456, 0.406],
            std=[0.229] if cfg.GRAYSCALE else [0.229, 0.224, 0.225]
        )

        return A.Compose(base_transforms + augmentations + [normalize, ToTensorV2()])

    else:  # validation
        # Только нормализация для валидации
        normalize = A.Normalize(
            mean=[0.485] if cfg.GRAYSCALE else [0.485, 0.456, 0.406],
            std=[0.229] if cfg.GRAYSCALE else [0.229, 0.224, 0.225]
        )

        return A.Compose(base_transforms + [normalize, ToTensorV2()])


class TiltAugmentation:
    """Специальная аугментация наклона банки"""

    @staticmethod
    def apply_tilt(image, max_angle=3):
        """
        Применяет наклон к изображению банки
        Работает как с grayscale, так и с RGB изображениями
        """
        if max_angle == 0:
            return image

        angle = np.random.uniform(-max_angle, max_angle)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        tilted_image = cv2.warpAffine(
            image, rotation_matrix, (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        return tilted_image