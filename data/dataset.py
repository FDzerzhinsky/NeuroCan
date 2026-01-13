import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import re


class SodaCanDataset(Dataset):
    """Dataset для банок газировки с поддержкой grayscale.

    Параметры:
    - data_dir: папка с изображениями
    - labels_file: путь к файлу разметки (опционально). Если None, автоматически ищем labels.txt или values.txt в data_dir.
    """

    def __init__(self, data_dir, labels_file=None, phase='train', transform=None):
        self.data_dir = Path(data_dir)
        self.phase = phase
        self.transform = transform

        # Определяем файл разметки
        if labels_file is None:
            # Проверяем в самой папке data_dir
            candidate1 = self.data_dir / 'labels.txt'
            candidate2 = self.data_dir / 'values.txt'
            if candidate1.exists():
                labels_file = candidate1
            elif candidate2.exists():
                labels_file = candidate2
            else:
                # Если не найдено — пробуем глобально настроенный путь (оставляем старую совместимость)
                from config.config import cfg
                if cfg.LABELS_FILE.exists():
                    labels_file = cfg.LABELS_FILE
                else:
                    raise FileNotFoundError(f"Labels file not found in {self.data_dir}. Expected 'labels.txt' or 'values.txt'.")

        self.labels_file = Path(labels_file)
        self.samples = self._load_labels(self.labels_file)
        # Дополнительная структура: список дополнительно сгенерированных записей
        # Каждая запись: (orig_idx, aug_type, params)
        # aug_type: one of 'tilt','vertical_shift','color','gauss_noise','motion_blur','median_blur'
        self.augmented_entries = []

        # Примечание: генерация дополнительных записей выполняется отдельно
        # через метод `generate_augmented_entries_for_indices`, чтобы можно было
        # создавать такие записи только для обучающей части после random_split.

        # Импортируем здесь чтобы избежать циклических импортов
        from data.transforms import TiltAugmentation
        self.tilt_augmentation = TiltAugmentation() if phase == 'train' else None

        print(f"Loaded {len(self.samples)} samples for {phase} (labels: {self.labels_file})")

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

        if not samples:
            raise RuntimeError(f"No samples loaded from labels file {labels_file}. Check your dataset and labels.")

        # Дополнительная проверка: выводим информацию об углах
        angles = [angle for _, angle in samples]
        if angles:
            print(f"Angle range: {min(angles)}-{max(angles)}")
            print(f"Unique angles: {len(set(angles))}")

        return samples

    def _generate_augmented_entries(self):
        """(internal) Генерирует дополнительные аугментированные записи на основе конфигурации
        для полного набора образцов (deprecated — используйте generate_augmented_entries_for_indices).
        """
        from config.config import cfg
        import math

        original_count = len(self.samples)

        def pct_to_count(v):
            """Интерпретирует v как процент или долю и возвращает количество образцов (ceil).
            Если v>1 — считаем как процент (0..100), иначе как долю (0..1).
            """
            try:
                val = float(v)
            except Exception:
                return 0
            if val <= 0:
                return 0
            if val > 1:
                # treated as percent
                return int(math.ceil(original_count * (val / 100.0)))
            else:
                return int(math.ceil(original_count * val))

        # 1. Tilt Augmentation
        count_tilt = pct_to_count(getattr(cfg, 'AUG_ADD_PCT_TILT', getattr(cfg, 'AUG_P_TILT', 0)))
        if count_tilt > 0:
            replace = count_tilt > original_count
            idxs = list(np.random.choice(np.arange(original_count), size=count_tilt, replace=replace))
            for orig in idxs:
                self.augmented_entries.append((int(orig), 'tilt', {}))

        # 2. Vertical Shift (VERTICAL_SHIFT_PERCENT treated as percent/delta like others)
        count_shift = pct_to_count(getattr(cfg, 'AUG_ADD_PCT_VERTICAL', getattr(cfg, 'AUG_P_VERTICAL', getattr(cfg, 'VERTICAL_SHIFT_PERCENT', 0))))
        if count_shift > 0:
            replace = count_shift > original_count
            idxs = list(np.random.choice(np.arange(original_count), size=count_shift, replace=replace))
            for orig in idxs:
                self.augmented_entries.append((int(orig), 'vertical_shift', {}))

        # 3. Color Augmentation
        count_color = pct_to_count(getattr(cfg, 'AUG_ADD_PCT_COLOR', getattr(cfg, 'AUG_P_COLOR', 0)))
        if count_color > 0:
            replace = count_color > original_count
            idxs = list(np.random.choice(np.arange(original_count), size=count_color, replace=replace))
            for orig in idxs:
                self.augmented_entries.append((int(orig), 'color', {}))

        # 4. Noise/Blur group: split AUG_P_NOISE between gauss_noise, motion_blur, median_blur evenly
        count_noise_total = pct_to_count(getattr(cfg, 'AUG_ADD_PCT_NOISE', getattr(cfg, 'AUG_P_NOISE', 0)))
        if count_noise_total > 0:
            # split approximately equally
            sub = count_noise_total // 3
            remainder = count_noise_total - sub * 3
            counts = [sub, sub, sub]
            for i in range(remainder):
                counts[i] += 1
            # gauss
            replace = counts[0] > original_count
            idxs = list(np.random.choice(np.arange(original_count), size=counts[0], replace=replace))
            for orig in idxs:
                self.augmented_entries.append((int(orig), 'gauss_noise', {}))
            # motion
            replace = counts[1] > original_count
            idxs = list(np.random.choice(np.arange(original_count), size=counts[1], replace=replace))
            for orig in idxs:
                self.augmented_entries.append((int(orig), 'motion_blur', {}))
            # median
            replace = counts[2] > original_count
            idxs = list(np.random.choice(np.arange(original_count), size=counts[2], replace=replace))
            for orig in idxs:
                self.augmented_entries.append((int(orig), 'median_blur', {}))

        print(f"Generated {len(self.augmented_entries)} augmented entries")
        total = original_count + len(self.augmented_entries)
        print(f"Augmentation summary (full dataset): originals={original_count}, generated={len(self.augmented_entries)}, total={total}")

    def generate_augmented_entries_for_indices(self, indices):
        """Генерирует augmented_entries только для выбранного набора индексов (indices: iterable of original indices).
        Это предотвращает утечку данных в валидацию: вызов производить только для train_dataset.indices после random_split.
        """
        # Очистим текущие дополнительные записи и сгенерируем новые
        self.augmented_entries = []
        from config.config import cfg
        import math

        original_count = len(indices)
        if original_count == 0:
            return

        def pct_to_count(v):
            try:
                val = float(v)
            except Exception:
                return 0
            if val <= 0:
                return 0
            if val > 1:
                return int(math.ceil(original_count * (val / 100.0)))
            else:
                return int(math.ceil(original_count * val))

        # helper to sample from provided indices
        def sample_indices(k, replace=False):
            return list(np.random.choice(np.array(indices), size=k, replace=replace))

        # 1. Tilt
        count_tilt = pct_to_count(getattr(cfg, 'AUG_ADD_PCT_TILT', getattr(cfg, 'AUG_P_TILT', 0)))
        if count_tilt > 0:
            replace = count_tilt > original_count
            idxs = sample_indices(count_tilt, replace=replace)
            for orig in idxs:
                self.augmented_entries.append((int(orig), 'tilt', {}))

        # 2. Vertical shift
        count_shift = pct_to_count(getattr(cfg, 'AUG_ADD_PCT_VERTICAL', getattr(cfg, 'AUG_P_VERTICAL', getattr(cfg, 'VERTICAL_SHIFT_PERCENT', 0))))
        if count_shift > 0:
            replace = count_shift > original_count
            idxs = sample_indices(count_shift, replace=replace)
            for orig in idxs:
                self.augmented_entries.append((int(orig), 'vertical_shift', {}))

        # 3. Color
        count_color = pct_to_count(getattr(cfg, 'AUG_ADD_PCT_COLOR', getattr(cfg, 'AUG_P_COLOR', 0)))
        if count_color > 0:
            replace = count_color > original_count
            idxs = sample_indices(count_color, replace=replace)
            for orig in idxs:
                self.augmented_entries.append((int(orig), 'color', {}))

        # 4. Noise/Blur group — split AUG_P_NOISE
        count_noise_total = pct_to_count(getattr(cfg, 'AUG_ADD_PCT_NOISE', getattr(cfg, 'AUG_P_NOISE', 0)))
        if count_noise_total > 0:
            sub = count_noise_total // 3
            remainder = count_noise_total - sub * 3
            counts = [sub, sub, sub]
            for i in range(remainder):
                counts[i] += 1
            idxs = sample_indices(counts[0], replace=(counts[0] > original_count))
            for orig in idxs:
                self.augmented_entries.append((int(orig), 'gauss_noise', {}))
            idxs = sample_indices(counts[1], replace=(counts[1] > original_count))
            for orig in idxs:
                self.augmented_entries.append((int(orig), 'motion_blur', {}))
            idxs = sample_indices(counts[2], replace=(counts[2] > original_count))
            for orig in idxs:
                self.augmented_entries.append((int(orig), 'median_blur', {}))

        print(f"Generated {len(self.augmented_entries)} augmented entries for indices (len={len(indices)})")
        total = len(self.samples) + len(self.augmented_entries)
        print(f"Augmentation summary (subset): originals={len(indices)}, generated={len(self.augmented_entries)}, total={total}")

    def __len__(self):
        return len(self.samples) + len(self.augmented_entries)

    def __getitem__(self, idx):
        from config.config import cfg

        # Если индекс относится к augmented_entries — обрабатываем отдельно
        if idx >= len(self.samples):
            aug_idx = idx - len(self.samples)
            orig_idx, aug_type, aug_params = self.augmented_entries[aug_idx]
            if aug_params is None:
                aug_params = {}
            img_path, angle = self.samples[orig_idx]
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {img_path}")
            image = np.expand_dims(image, axis=-1)

            # Применяем именно указанную аугментацию детерминированно
            if aug_type == 'tilt':
                from data.transforms import TiltAugmentation
                # if params absent, choose random angle
                if 'angle' in aug_params and aug_params['angle'] is not None:
                    ang = float(aug_params.get('angle'))
                else:
                    ang = float(np.random.uniform(-getattr(cfg, 'MAX_TILT_ANGLE', 0), getattr(cfg, 'MAX_TILT_ANGLE', 0)))
                image = TiltAugmentation.apply_tilt(image, cfg.MAX_TILT_ANGLE, angle=ang)
            elif aug_type == 'vertical_shift':
                if 'shift_percent' in aug_params and aug_params.get('shift_percent') is not None:
                    shift_percent = float(aug_params.get('shift_percent'))
                else:
                    # choose random signed shift within allowed percent
                    ms = getattr(cfg, 'VERTICAL_SHIFT_PERCENT', 0.0)
                    shift_percent = float(np.random.uniform(-ms, ms))
                shift_pixels = int(round(shift_percent * image.shape[0]))
                M = np.float32([[1, 0, 0], [0, 1, shift_pixels]])
                image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            elif aug_type == 'color':
                # if specific params provided, use them; else pick randomly like in train transforms
                if 'method' in aug_params and aug_params.get('method') == 'gamma' and 'gamma' in aug_params:
                    g = float(aug_params.get('gamma'))
                    arr = image.astype(np.float32) / 255.0
                    arr = np.power(arr, g)
                    image = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                elif 'method' in aug_params and aug_params.get('method') == 'brightness_contrast' and ('brightness' in aug_params or 'contrast' in aug_params):
                    b = float(aug_params.get('brightness', 0.0))
                    c = float(aug_params.get('contrast', 0.0))
                    beta = int(round(b * 255.0))
                    alpha = 1.0 + c
                    image = np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)
                else:
                    # choose randomly
                    if np.random.rand() < 0.5:
                        from data.transforms import apply_gamma_random
                        image, params = apply_gamma_random(image)
                    else:
                        from data.transforms import apply_brightness_contrast_random
                        image, params = apply_brightness_contrast_random(image)
            elif aug_type == 'gauss_noise':
                if 'sigma' in aug_params and aug_params.get('sigma') is not None:
                    sigma = float(aug_params.get('sigma'))
                else:
                    from data.transforms import apply_gauss_noise_random
                    image, params = apply_gauss_noise_random(image)
                    # params may be ignored
            elif aug_type == 'motion_blur':
                if 'ksize' in aug_params and aug_params.get('ksize') is not None:
                    k = int(aug_params.get('ksize'))
                else:
                    from data.transforms import apply_motion_blur_random
                    image, params = apply_motion_blur_random(image)
            elif aug_type == 'median_blur':
                if 'ksize' in aug_params and aug_params.get('ksize') is not None:
                    k = int(aug_params.get('ksize'))
                else:
                    from data.transforms import apply_median_blur_random
                    image, params = apply_median_blur_random(image)

            # После применения детерминированной аугментации — применяем общую transform (Normalization/ToTensor)
            if self.transform:
                image = self.transform(image=image)['image']

            # Преобразуем угол в циклические координаты
            angle_rad = torch.tensor(angle) * 2 * torch.pi / 360
            sin_target = torch.sin(angle_rad)
            cos_target = torch.cos(angle_rad)

            out = {
                'image': image,
                'angle': angle,
                'sin_target': sin_target,
                'cos_target': cos_target,
                'file_path': str(img_path),
                'augmented': True,
                'aug_type': aug_type if aug_type is not None else '',
                'aug_params': aug_params if aug_params is not None else {}
            }
            # Validation: ensure no None values that would break default_collate
            if out['image'] is None:
                raise ValueError(f"Dataset __getitem__ produced empty image for augmented idx {idx}, path={img_path}, aug_type={aug_type}")
            return out

        # --- ORIGINAL PATH (non-augmented sample) ---
        img_path, angle = self.samples[idx]

        # Загружаем изображение
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)  # Сразу загружаем как grayscale

        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {img_path}")

        # Добавляем канальное измерение (H, W) -> (H, W, 1)
        image = np.expand_dims(image, axis=-1)

        # Применяем специальную аугментацию наклона
        if self.phase == 'train' and self.tilt_augmentation:
            # Применяем тильт с вероятностью cfg.AUG_P_TILT (по умолчанию 0.8)
            p_tilt = getattr(cfg, 'AUG_P_TILT', 1.0)
            if np.random.rand() < float(p_tilt):
                image = self.tilt_augmentation.apply_tilt(image, cfg.MAX_TILT_ANGLE)

        # Применяем стандартные трансформы
        if self.transform:
            image = self.transform(image=image)['image']

        # Преобразуем угол в циклические координаты
        angle_rad = torch.tensor(angle) * 2 * torch.pi / 360
        sin_target = torch.sin(angle_rad)
        cos_target = torch.cos(angle_rad)

        out = {
            'image': image,
            'angle': angle,
            'sin_target': sin_target,
            'cos_target': cos_target,
            'file_path': str(img_path),
            'augmented': False,
            'aug_type': '',
            'aug_params': {}
        }
        if out['image'] is None:
            raise ValueError(f"Dataset __getitem__ produced empty image for idx {idx}, path={img_path}")
        return out
