import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import warnings

# Фильтруем конкретные предупреждения albumentations
warnings.filterwarnings("ignore", message="The image is already gray.", category=UserWarning)

from config.config import cfg


# Вспомогательная функция: вертикальный сдвиг изображения на случайный процент
def _vertical_shift(image, max_shift_percent=None):
    """Смещает изображение строго по вертикали на случайный процент высоты в диапазоне [-max_shift_percent, max_shift_percent].
    Горизонтальное смещение отсутствует.
    """
    if max_shift_percent is None:
        max_shift_percent = cfg.VERTICAL_SHIFT_PERCENT
    if max_shift_percent == 0:
        return image
    h, w = image.shape[:2]
    shift_pixels = int(np.round(np.random.uniform(-max_shift_percent, max_shift_percent) * h))
    # Параметр матрицы сдвига по вертикали
    M = np.float32([[1, 0, 0], [0, 1, shift_pixels]])
    shifted = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return shifted


def _vert_wrapper(img, **kwargs):
    return _vertical_shift(img, max_shift_percent=cfg.VERTICAL_SHIFT_PERCENT)


# === Новые вспомогательные функции для детерминированного/рандомного применения отдельных аугментаций ===
def apply_gamma_random(image, gmin=None, gmax=None):
    """Применяет случайный gamma из диапазона cfg.GAMMA_LIMIT. Возвращает (img, params)."""
    if gmin is None or gmax is None:
        gmin, gmax = getattr(cfg, 'GAMMA_LIMIT', (80, 120))
    gamma = float(np.random.uniform(gmin / 100.0, gmax / 100.0))
    arr = image.astype(np.float32) / 255.0
    arr = np.power(arr, gamma)
    after = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return after, {'method': 'gamma', 'gamma': float(gamma)}


def apply_brightness_contrast_random(image, b_limit=None, c_limit=None):
    """Применяет случайную яркость/контраст в пределах лимитов. Возвращает (img, params)."""
    if b_limit is None:
        b_limit = getattr(cfg, 'BRIGHTNESS_LIMIT', 0.1)
    if c_limit is None:
        c_limit = getattr(cfg, 'CONTRAST_LIMIT', 0.1)
    brightness = float(np.random.uniform(-b_limit, b_limit))
    contrast = float(np.random.uniform(-c_limit, c_limit))
    beta = int(round(brightness * 255.0))
    alpha = 1.0 + contrast
    after = np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    return after, {'method': 'brightness_contrast', 'brightness': float(brightness), 'contrast': float(contrast)}


def apply_gauss_noise_random(image, var_min=None, var_max=None):
    if var_min is None or var_max is None:
        var_min, var_max = getattr(cfg, 'GAUSS_NOISE_VAR', (10.0, 50.0))
    sigma = float(np.random.uniform(var_min, var_max))
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    after = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return after, {'sigma': float(sigma)}


def apply_motion_blur_random(image, limit=None):
    if limit is None:
        limit = int(getattr(cfg, 'MOTION_BLUR_LIMIT', 3))
    limit = max(1, int(limit))
    k = int(np.random.randint(1, limit + 1))
    if k % 2 == 0:
        k = max(1, k - 1)
    if k <= 1:
        return image, {'ksize': int(k)}
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = np.ones(k, dtype=np.float32)
    kernel = kernel / k
    after = cv2.filter2D(image, -1, kernel)
    return after, {'ksize': int(k)}


def apply_median_blur_random(image, limit=None):
    if limit is None:
        limit = int(getattr(cfg, 'MEDIAN_BLUR_LIMIT', 3))
    limit = max(1, int(limit))
    k = int(np.random.randint(1, limit + 1))
    if k % 2 == 0:
        k = max(1, k - 1)
    if k <= 1:
        return image, {'ksize': int(k)}
    ch = image[:, :, 0] if image.ndim == 3 else image
    res = cv2.medianBlur(ch, k)
    after = np.expand_dims(res, axis=-1)
    return after, {'ksize': int(k)}


def apply_color_random(image):
    """Выбирает либо gamma либо brightness/contrast, применяет и возвращает (img, params)."""
    if np.random.rand() < 0.5:
        return apply_gamma_random(image)
    else:
        return apply_brightness_contrast_random(image)


# === Продолжение: get_transforms и TiltAugmentation как было ===
def get_transforms(phase='train'):
    """
    Возвращает трансформы для обучения/валидации
    Теперь поддерживает grayscale изображения без лишних предупреждений
    """
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

            # Горизонтальные смещения и поворот не применяются здесь — единственный источник тильта находится в dataset.TiltAugmentation
            # Отдельная, строго вертикальная трансформация с контролируемой амплитудой.
            # Применяется с вероятностью p_vertical (берётся из cfg).
            A.Lambda(image=_vert_wrapper, p=cfg.AUG_P_VERTICAL),

            A.OneOf([
                A.RandomGamma(gamma_limit=cfg.GAMMA_LIMIT, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=cfg.BRIGHTNESS_LIMIT,
                    contrast_limit=cfg.CONTRAST_LIMIT,
                    p=0.5
                ),
            ], p=cfg.AUG_P_COLOR),
            A.OneOf([
                A.GaussNoise(var_limit=cfg.GAUSS_NOISE_VAR),
                A.MotionBlur(blur_limit=cfg.MOTION_BLUR_LIMIT),
                A.MedianBlur(blur_limit=cfg.MEDIAN_BLUR_LIMIT),
            ], p=cfg.AUG_P_NOISE),
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
    def apply_tilt(image, max_angle=None, angle: float = None):
        """
        Применяет наклон (поворот) к изображению банки.
        Если параметр angle задан (в градусах), применяется именно он; иначе выбирается случайный угол в диапазоне [-max_angle, max_angle].
        Работает как с grayscale, так и с RGB изображениями.
        """
        if max_angle is None:
            max_angle = cfg.MAX_TILT_ANGLE

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

