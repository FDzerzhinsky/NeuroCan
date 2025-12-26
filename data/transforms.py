import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import warnings

# Фильтруем конкретные предупреждения albumentations
warnings.filterwarnings("ignore", message="The image is already gray.", category=UserWarning)


# Значение вертикального сдвига (процент от высоты), экспортируемое для тестов
VERTICAL_SHIFT_PERCENT = 0.02


# Вспомогательная функция: вертикальный сдвиг изображения на случайный процент
def _vertical_shift(image, max_shift_percent=VERTICAL_SHIFT_PERCENT):
    """Смещает изображение строго по вертикали на случайный процент высоты в диапазоне [-max_shift_percent, max_shift_percent].
    Горизонтальное смещение отсутствует.
    """
    if max_shift_percent == 0:
        return image
    h, w = image.shape[:2]
    shift_pixels = int(np.round(np.random.uniform(-max_shift_percent, max_shift_percent) * h))
    # Параметр матрицы сдвига по вертикали
    M = np.float32([[1, 0, 0], [0, 1, shift_pixels]])
    shifted = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return shifted


def _vert_wrapper(img, **kwargs):
    return _vertical_shift(img, max_shift_percent=VERTICAL_SHIFT_PERCENT)


def get_transforms(phase='train'):
    """
    Возвращает трансформы для обучения/валидации
    Теперь поддерживает grayscale изображения без лишних предупреждений
    """
    from config.config import cfg

    # Базовые трансформы которые применяются всегда
    base_transforms = []

    if phase == 'train':
        # Аугментации для обучения
        augmentations = [
            # ORIGINAL:
            # A.Affine(
            #     rotate=(-cfg.MAX_TILT_ANGLE, cfg.MAX_TILT_ANGLE),
            #     translate_percent=(-0.02, 0.02),
            #     scale=(0.98, 1.02),
            #     shear=(-1, 1),
            #     p=0.8
            # ),

            # Новая конфигурация A.Affine: поворот и shear оставляем, масштабирование отключено (1.0),
            # горизонтальные смещения запрещены (translate_percent по горизонтали = 0).
            # A.Affine(
            #     rotate=(-cfg.MAX_TILT_ANGLE, cfg.MAX_TILT_ANGLE),
            #     # translate_percent intentionally omitted to avoid internal formatting issues;
            #     # вертикальные сдвиги контролируются отдельно via _vertical_shift
            #     # horizontal translation must be forbidden per requirements
#
#                 scale=(1.0, 1.0),  # масштабирование отключено
#                 shear=(-1, 1),
#                 p=0.8
#             ),
            # Горизонтальные смещения и поворот не применяются здесь — единственный источник тильта находится в dataset.TiltAugmentation
            # Отдельная, строго вертикальная трансформация с контролируемой амплитудой.
            # Применяется с вероятностью p_vertical (совпадает с оригинальной вероятностью 0.8).
            A.Lambda(image=_vert_wrapper, p=0.8),

            A.OneOf([
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.5
                ),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(),
                A.MotionBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
            ], p=0.3),
        ]

        # Нормализация для grayscale
        normalize = A.Normalize(
            mean=(0.485,),
            std=(0.229,)
        )

        return A.Compose(base_transforms + augmentations + [normalize, ToTensorV2()])

    else:  # validation
        # Только нормализация для валидации
        normalize = A.Normalize(
            mean=(0.485,),
            std=(0.229,)
        )

        return A.Compose(base_transforms + [normalize, ToTensorV2()])


class TiltAugmentation:
    """Специальная аугментация наклона банки"""

    @staticmethod
    def apply_tilt(image, max_angle=1.5, angle: float = None):
        """
        Применяет наклон (поворот) к изображению банки.
        Если параметр angle задан (в градусах), применяется именно он; иначе выбирается случайный угол в диапазоне [-max_angle, max_angle].
        Работает как с grayscale, так и с RGB изображениями.
        """
        if max_angle == 0 and angle is None:
            return image

        if angle is None:
            angle = float(np.random.uniform(-max_angle, max_angle))

        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        tilted_image = cv2.warpAffine(
            image, rotation_matrix, (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        return tilted_image

