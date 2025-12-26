import sys
from pathlib import Path
import cv2
import numpy as np
import torch

# Убедимся, что корень проекта в sys.path, чтобы можно было импортировать локальные модули (`data`, `config` и т.д.)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.transforms import get_transforms, TiltAugmentation, VERTICAL_SHIFT_PERCENT
from config.config import cfg

OUT_DIR = Path('logs') / 'aug_test'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _make_gradient(h=256, w=256):
    # Вертикальный градиент (значение изменяется по вертикали), чтобы вертикальные сдвиги были заметны
    g = np.tile(np.linspace(0, 255, h, dtype=np.uint8).reshape(h, 1), (1, w))
    return g


def _make_circle(h=256, w=256):
    # Рисуем светлый прямоугольник на тёмном фоне (используем монохромный образ)
    img = np.full((h, w), 128, dtype=np.uint8)
    # Координаты прямоугольника: от четверти до трёх четвертей по обеим осям
    tl = (w // 4, h // 4)
    br = (3 * w // 4, 3 * h // 4)
    cv2.rectangle(img, tl, br, int(255), -1)
    return img


def _save_uint8(img, path):
    # img: HxW or HxWx1 or HxWx3 uint8
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    cv2.imwrite(str(path), img)


def _denormalize_tensor_to_uint8(tensor, mean=0.485, std=0.229):
    # tensor: torch.Tensor (C,H,W)
    arr = tensor.detach().cpu().numpy()
    # C,H,W -> H,W,C
    arr = arr.transpose(1, 2, 0)
    arr = arr * std + mean
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    # If single channel, squeeze
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr


def test_apply_and_visualize_transforms():
    """Применяем аугментации к синтетическим изображениям, сохраняем и делаем базовые проверки."""
    train_tf = get_transforms('train')
    val_tf = get_transforms('val')
    tilt = TiltAugmentation()

    samples = {
        'gradient': _make_gradient(),
        'circle': _make_circle()
    }

    for name, img in samples.items():
        h, w = img.shape
        img_hwc = np.expand_dims(img, axis=-1)  # H,W,1

        # Сохраним оригинал
        _save_uint8(img_hwc, OUT_DIR / f'{name}_orig.png')

        # --- Детерминированные крайние варианты для каждой аугментации ---
        # 1) Tilt: детерминированные крайние значения (-MAX, +MAX) по cfg
        # используем метод из основного пайплайна для детерминированного поворота
        angle_min = -cfg.MAX_TILT_ANGLE
        angle_max = cfg.MAX_TILT_ANGLE
        min_tilt = TiltAugmentation.apply_tilt(img_hwc.copy(), max_angle=cfg.MAX_TILT_ANGLE, angle=angle_min)
        max_tilt = TiltAugmentation.apply_tilt(img_hwc.copy(), max_angle=cfg.MAX_TILT_ANGLE, angle=angle_max)
        _save_uint8(min_tilt, OUT_DIR / f'{name}_tilt_min.png')
        _save_uint8(max_tilt, OUT_DIR / f'{name}_tilt_max.png')

        # 2) Vertical shift: shifts by ±max_shift_percent (тот же, что в transforms)
        max_shift_percent = VERTICAL_SHIFT_PERCENT
        def vert_shift_fixed(img, shift_percent):
            h0,w0 = img.shape[:2]
            shift_pixels = int(round(shift_percent * h0))
            M = np.float32([[1,0,0],[0,1,shift_pixels]])
            return cv2.warpAffine(img, M, (w0,h0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        vs_min = vert_shift_fixed(img_hwc.copy(), -max_shift_percent)
        vs_max = vert_shift_fixed(img_hwc.copy(), max_shift_percent)
        _save_uint8(vs_min, OUT_DIR / f'{name}_vert_min.png')
        _save_uint8(vs_max, OUT_DIR / f'{name}_vert_max.png')

        # 3) Gamma: min and max taken из transforms (gamma_limit=(80,120) -> 0.8-1.2)
        gamma_min = 80.0 / 100.0
        gamma_max = 120.0 / 100.0
        def gamma_adjust(img, gamma):
            arr = img.astype(np.float32) / 255.0
            arr = np.power(arr, gamma)
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            return arr

        g_min = gamma_adjust(img_hwc.copy(), gamma_min)
        g_max = gamma_adjust(img_hwc.copy(), gamma_max)
        _save_uint8(g_min, OUT_DIR / f'{name}_gamma_min.png')
        _save_uint8(g_max, OUT_DIR / f'{name}_gamma_max.png')

        # 4) Brightness/Contrast extremes: limits taken from transforms (±0.1)
        def brightness_contrast(img, brightness, contrast):
            # brightness in [-0.1,0.1] means beta = 255*brightness
            beta = int(round(brightness * 255.0))
            alpha = 1.0 + contrast
            res = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
            return res

        bc_min = brightness_contrast(img_hwc.copy(), -0.1, -0.1)
        bc_max = brightness_contrast(img_hwc.copy(), 0.1, 0.1)
        _save_uint8(bc_min, OUT_DIR / f'{name}_bc_min.png')
        _save_uint8(bc_max, OUT_DIR / f'{name}_bc_max.png')

        # 5) Gauss noise: limits correspond to original code var_limit approx (10.0-50.0)
        def gauss_noise(img, sigma):
            noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
            res = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            return res

        gn_min = gauss_noise(img_hwc.copy(), 10.0)
        gn_max = gauss_noise(img_hwc.copy(), 50.0)
        _save_uint8(gn_min, OUT_DIR / f'{name}_gauss_min.png')
        _save_uint8(gn_max, OUT_DIR / f'{name}_gauss_max.png')

        # 6) Motion blur: kernel sizes 1 (none) and 3
        def motion_blur(img, k):
            if k <= 1:
                return img
            # horizontal motion blur kernel
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[k//2, :] = np.ones(k, dtype=np.float32)
            kernel = kernel / k
            res = cv2.filter2D(img, -1, kernel)
            return res

        mb_min = motion_blur(img_hwc.copy(), 1)
        mb_max = motion_blur(img_hwc.copy(), 3)
        _save_uint8(mb_min, OUT_DIR / f'{name}_motion_min.png')
        _save_uint8(mb_max, OUT_DIR / f'{name}_motion_max.png')

        # 7) Median blur: ksize 1 and 3
        def median_blur(img, k):
            if k <= 1:
                return img
            # cv2.medianBlur expects single channel
            ch = img[:, :, 0] if img.ndim == 3 else img
            res = cv2.medianBlur(ch, k)
            if img.ndim == 3:
                return np.expand_dims(res, axis=-1)
            return res

        mb2_min = median_blur(img_hwc.copy(), 1)
        mb2_max = median_blur(img_hwc.copy(), 3)
        _save_uint8(mb2_min, OUT_DIR / f'{name}_median_min.png')
        _save_uint8(mb2_max, OUT_DIR / f'{name}_median_max.png')

        # Проверки: убедимся, что min/max отличаются
        assert not np.array_equal(min_tilt.squeeze(), max_tilt.squeeze()), 'tilt min/max same'
        # vertical shift может при определённых симметриях выглядеть одинаково при + и - смещении;
        # в таком редком случае достаточно убедиться, что хотя бы одна из версий отличается от оригинала
        orig = img_hwc.squeeze()
        if np.array_equal(vs_min.squeeze(), vs_max.squeeze()):
            assert (not np.array_equal(vs_min.squeeze(), orig)) or (not np.array_equal(vs_max.squeeze(), orig)), 'vertical shift did not change image'
        else:
            assert True
        assert not np.array_equal(g_min.squeeze(), g_max.squeeze()), 'gamma min/max same'
        assert not np.array_equal(bc_min.squeeze(), bc_max.squeeze()), 'bc min/max same'
        assert not np.array_equal(gn_min.squeeze(), gn_max.squeeze()), 'gauss min/max same'
        # motion blur и median blur могут не менять простой вертикальный градиент (зависит от ориентации ядра).
        # Поэтому здесь не требуем строгого отличия; изображения сохранены для визуальной проверки.

        # 1) TiltAugmentation несколько раз
        tilted_images = []
        for i in range(3):
            t = tilt.apply_tilt(img_hwc.copy(), max_angle=cfg.MAX_TILT_ANGLE)
            tilted_images.append(t)
            _save_uint8(t, OUT_DIR / f'{name}_tilt_{i}.png')

        # Проверка: каждый результат имеет форму H,W,1
        for t in tilted_images:
            assert t.shape[0] == h and t.shape[1] == w
            assert (t.ndim == 3 and t.shape[2] == 1) or (t.ndim == 2)

        # 2) Albumentations pipeline (train) несколько раз
        transformed_tensors = []
        for i in range(4):
            out = train_tf(image=img_hwc)
            tensor = out['image']  # torch tensor C,H,W
            assert isinstance(tensor, torch.Tensor)
            assert tensor.ndim == 3
            assert tensor.shape[0] == 1  # one channel
            arr = _denormalize_tensor_to_uint8(tensor)
            transformed_tensors.append(arr)
            _save_uint8(arr, OUT_DIR / f'{name}_train_tf_{i}.png')

        # 3) Albumentations pipeline (val) deterministically
        out_val = val_tf(image=img_hwc)
        tensor_val = out_val['image']
        arr_val = _denormalize_tensor_to_uint8(tensor_val)
        _save_uint8(arr_val, OUT_DIR / f'{name}_val_tf.png')

        # Базовые assert'ы
        # - трансформированный вал должен иметь ту же форму
        assert arr_val.shape[0] == h and arr_val.shape[1] == w

        # - среди train-результатов должны быть отличающиеся варианты
        unique_hashes = set()
        for arr in transformed_tensors:
            unique_hashes.add(arr.tobytes())
        assert len(unique_hashes) >= 2, "Ожидалось минимум 2 уникальных аугментированных изображения"

        # - по крайней мере один tilt результат должен отличаться от оригинала
        orig_bytes = img_hwc.squeeze().tobytes()
        tilt_diff = any(t.squeeze().tobytes() != orig_bytes for t in tilted_images)
        assert tilt_diff, "Ожидалось, что TiltAugmentation изменит изображение"

    print(f"Saved augmentations outputs to: {OUT_DIR}")
