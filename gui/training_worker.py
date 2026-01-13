from PySide6.QtCore import QThread, Signal
from data.dataset import SodaCanDataset
from data.transforms import get_transforms
from training.trainer import CanRotationTrainer
from config.config import cfg
from pathlib import Path

import numpy as np
import cv2
import albumentations as A


class TrainingWorker(QThread):
    # epoch, train_loss, val_loss, train_acc, val_acc
    epoch_signal = Signal(int, float, float, float, float)
    # Передаём примеры аугментаций (словарь: type-> {'before':ndarray, 'after':ndarray, 'params': dict})
    augmentations_signal = Signal(object)

    def __init__(self, dataset_path, epochs, batch_size, lr, parent=None):
        super().__init__(parent)
        self.dataset_path = Path(dataset_path)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self._stop_requested = False
        self.trainer = None

    def _generate_aug_examples(self, dataset, max_examples=6, preview_only: bool = False):
        """
        Генерируем примеры аугментаций используя настройки из cfg.
        Если preview_only=True — проигрываем режим "холостой генерации":
          - вероятность каждой аугментации считается как 1 (все типы генерируются),
          - параметры выставляются в максимальные значения из cfg (не случайные).
        Возвращаем dict: ключ — строка типа аугментации, значение — dict {'before','after','params'}.
        """
        examples = {}
        n_samples = len(dataset)
        if n_samples == 0:
            return examples
        idxs = np.random.choice(np.arange(n_samples), size=min(max_examples, n_samples), replace=False)

        from data.transforms import TiltAugmentation

        for idx in idxs:
            try:
                img_path, _ = dataset.samples[idx]
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img_hwc = np.expand_dims(img, axis=-1)
                h, w = img_hwc.shape[:2]

                # PREVIEW MODE: force generation and max params
                if preview_only:
                    # 1) Tilt (force if configured)
                    if 'tilt' not in examples and getattr(cfg, 'MAX_TILT_ANGLE', 0) != 0:
                        max_angle = getattr(cfg, 'MAX_TILT_ANGLE', 0)
                        angle = float(max_angle)
                        after = TiltAugmentation.apply_tilt(img_hwc.copy(), max_angle=max_angle, angle=angle)
                        examples['tilt'] = {'before': img_hwc.copy(), 'after': after.astype(np.uint8), 'params': {'angle': float(angle)}}

                    # 2) Vertical shift (force)
                    if 'vertical_shift' not in examples and getattr(cfg, 'VERTICAL_SHIFT_PERCENT', 0.0) != 0:
                        max_shift_percent = getattr(cfg, 'VERTICAL_SHIFT_PERCENT', 0.0)
                        shift_percent = float(max_shift_percent)
                        shift_pixels = int(round(shift_percent * h))
                        M = np.float32([[1, 0, 0], [0, 1, shift_pixels]])
                        after = cv2.warpAffine(img_hwc.copy(), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                        examples['vertical_shift'] = {'before': img_hwc.copy(), 'after': after.astype(np.uint8), 'params': {'shift_percent': float(shift_percent), 'shift_pixels': int(shift_pixels)}}

                    # 3) Color (use gamma max)
                    if 'color' not in examples:
                        gmin, gmax = getattr(cfg, 'GAMMA_LIMIT', (80, 120))
                        gamma = float(gmax / 100.0)
                        arr = img_hwc.astype(np.float32) / 255.0
                        arr = np.power(arr, gamma)
                        after = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                        examples['color'] = {'before': img_hwc.copy(), 'after': after, 'params': {'method': 'gamma', 'gamma': float(gamma)}}

                    # 4a) Gauss noise (max sigma)
                    if 'gauss_noise' not in examples:
                        gmin, gmax = getattr(cfg, 'GAUSS_NOISE_VAR', (10.0, 50.0))
                        sigma = float(gmax)
                        noise = np.random.normal(0, sigma, img_hwc.shape).astype(np.float32)
                        after = np.clip(img_hwc.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                        examples['gauss_noise'] = {'before': img_hwc.copy(), 'after': after, 'params': {'sigma': float(sigma)}}

                    # 4b) Motion blur (max k)
                    if 'motion_blur' not in examples:
                        limit = max(1, int(getattr(cfg, 'MOTION_BLUR_LIMIT', 3)))
                        k = int(limit)
                        if k % 2 == 0:
                            k = max(1, k - 1)
                        if k <= 1:
                            after = img_hwc.copy()
                        else:
                            kernel = np.zeros((k, k), dtype=np.float32)
                            kernel[k // 2, :] = np.ones(k, dtype=np.float32)
                            kernel = kernel / k
                            after = cv2.filter2D(img_hwc, -1, kernel)
                        examples['motion_blur'] = {'before': img_hwc.copy(), 'after': after.astype(np.uint8), 'params': {'ksize': int(k)}}

                    # 4c) Median blur (max k)
                    if 'median_blur' not in examples:
                        limit = max(1, int(getattr(cfg, 'MEDIAN_BLUR_LIMIT', 3)))
                        k = int(limit)
                        if k % 2 == 0:
                            k = max(1, k - 1)
                        if k <= 1:
                            after = img_hwc.copy()
                        else:
                            ch = img_hwc[:, :, 0]
                            res = cv2.medianBlur(ch, k)
                            after = np.expand_dims(res, axis=-1)
                        examples['median_blur'] = {'before': img_hwc.copy(), 'after': after.astype(np.uint8), 'params': {'ksize': int(k)}}

                    # If got all desired types — break
                    if len(examples) >= 6:
                        break

                    # continue to next sample
                    continue

                # --- NORMAL TRAINING MODE (randomized) ---
                # 1) Tilt
                if 'tilt' not in examples and getattr(cfg, 'AUG_P_TILT', 0.0) > 0 and np.random.rand() < float(getattr(cfg, 'AUG_P_TILT', 0.0)):
                    max_angle = getattr(cfg, 'MAX_TILT_ANGLE', 0)
                    # выбранный угол — детерминируем здесь
                    angle = float(np.random.uniform(-max_angle, max_angle)) if max_angle != 0 else 0.0
                    after = TiltAugmentation.apply_tilt(img_hwc.copy(), max_angle=max_angle, angle=angle)
                    examples['tilt'] = {'before': img_hwc.copy(), 'after': after.astype(np.uint8), 'params': {'angle': float(angle)}}

                # 2) Vertical shift
                if 'vertical_shift' not in examples and getattr(cfg, 'AUG_P_VERTICAL', 0.0) > 0 and np.random.rand() < float(getattr(cfg, 'AUG_P_VERTICAL', 0.0)):
                    max_shift_percent = getattr(cfg, 'VERTICAL_SHIFT_PERCENT', 0.0)
                    if max_shift_percent != 0:
                        shift_percent = float(np.random.uniform(-max_shift_percent, max_shift_percent))
                        shift_pixels = int(round(shift_percent * h))
                        M = np.float32([[1, 0, 0], [0, 1, shift_pixels]])
                        after = cv2.warpAffine(img_hwc.copy(), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                        examples['vertical_shift'] = {'before': img_hwc.copy(), 'after': after.astype(np.uint8), 'params': {'shift_percent': float(shift_percent), 'shift_pixels': int(shift_pixels)}}

                # 3) Color (Gamma or Brightness/Contrast)
                if 'color' not in examples and getattr(cfg, 'AUG_P_COLOR', 0.0) > 0 and np.random.rand() < float(getattr(cfg, 'AUG_P_COLOR', 0.0)):
                    # выбирам либо gamma либо brightness/contrast
                    if np.random.rand() < 0.5:
                        # gamma: cfg.GAMMA_LIMIT — tuple (min,max) e.g. (80,120) -> 0.8..1.2
                        gmin, gmax = getattr(cfg, 'GAMMA_LIMIT', (80, 120))
                        gamma = float(np.random.uniform(gmin / 100.0, gmax / 100.0))
                        arr = img_hwc.astype(np.float32) / 255.0
                        arr = np.power(arr, gamma)
                        after = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                        examples['color'] = {'before': img_hwc.copy(), 'after': after, 'params': {'method': 'gamma', 'gamma': float(gamma)}}
                    else:
                        b_limit = getattr(cfg, 'BRIGHTNESS_LIMIT', 0.1)
                        c_limit = getattr(cfg, 'CONTRAST_LIMIT', 0.1)
                        brightness = float(np.random.uniform(-b_limit, b_limit))
                        contrast = float(np.random.uniform(-c_limit, c_limit))
                        beta = int(round(brightness * 255.0))
                        alpha = 1.0 + contrast
                        after = np.clip(alpha * img_hwc.astype(np.float32) + beta, 0, 255).astype(np.uint8)
                        examples['color'] = {'before': img_hwc.copy(), 'after': after, 'params': {'method': 'brightness_contrast', 'brightness': float(brightness), 'contrast': float(contrast)}}

                # 4) Noise / Blur
                if 'gauss_noise' not in examples and getattr(cfg, 'AUG_P_NOISE', 0.0) > 0 and np.random.rand() < float(getattr(cfg, 'AUG_P_NOISE', 0.0)):
                    r = np.random.rand()
                    if r < 0.4:
                        # Gauss noise: варьируем sigma
                        gmin, gmax = getattr(cfg, 'GAUSS_NOISE_VAR', (10.0, 50.0))
                        sigma = float(np.random.uniform(gmin, gmax))
                        noise = np.random.normal(0, sigma, img_hwc.shape).astype(np.float32)
                        after = np.clip(img_hwc.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                        examples['gauss_noise'] = {'before': img_hwc.copy(), 'after': after, 'params': {'sigma': float(sigma)}}
                    elif r < 0.7:
                        # Motion blur: choose odd kernel size between 3 and MOTION_BLUR_LIMIT
                        limit = max(1, int(getattr(cfg, 'MOTION_BLUR_LIMIT', 3)))
                        k = int(np.random.randint(1, limit + 1))
                        if k % 2 == 0:
                            k = max(1, k - 1)
                        if k <= 1:
                            after = img_hwc.copy()
                        else:
                            kernel = np.zeros((k, k), dtype=np.float32)
                            kernel[k // 2, :] = np.ones(k, dtype=np.float32)
                            kernel = kernel / k
                            after = cv2.filter2D(img_hwc, -1, kernel)
                        examples['motion_blur'] = {'before': img_hwc.copy(), 'after': after.astype(np.uint8), 'params': {'ksize': int(k)}}
                    else:
                        limit = max(1, int(getattr(cfg, 'MEDIAN_BLUR_LIMIT', 3)))
                        k = int(np.random.randint(1, limit + 1))
                        if k % 2 == 0:
                            k = max(1, k - 1)
                        if k <= 1:
                            after = img_hwc.copy()
                        else:
                            ch = img_hwc[:, :, 0]
                            res = cv2.medianBlur(ch, k)
                            after = np.expand_dims(res, axis=-1)
                        examples['median_blur'] = {'before': img_hwc.copy(), 'after': after.astype(np.uint8), 'params': {'ksize': int(k)}}

                # Если набрали все примеры — выйти
                if len(examples) >= 6:
                    break
            except Exception:
                continue

        return examples

    def run(self):
        # Ensure dirs
        cfg.ensure_dirs()

        # Определяем файл разметки в выбранной папке
        candidate1 = self.dataset_path / 'labels.txt'
        candidate2 = self.dataset_path / 'values.txt'
        labels_file = None
        if candidate1.exists():
            labels_file = candidate1
        elif candidate2.exists():
            labels_file = candidate2

        # Подготовка датасета, передаём labels_file (если None, датасет сам найдёт)
        transform = get_transforms('train')
        dataset = SodaCanDataset(self.dataset_path, labels_file=labels_file, phase='train', transform=transform)

        # Для MVP мы будем использовать DataLoader внутри тренера — тренер ожидает загрузчики
        # Здесь создаем train/val split
        from torch.utils.data import random_split, DataLoader
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_dataset.dataset.transform = get_transforms('train')
        val_dataset.dataset.transform = get_transforms('val')

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=cfg.NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=cfg.NUM_WORKERS)

        # Создаем модель/оптимизатор/scheduler
        from models.resnet_model import ResNetCanClassifier
        import torch
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        model = ResNetCanClassifier(num_classes=cfg.NUM_CLASSES)
        optimizer = AdamW(model.parameters(), lr=self.lr, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        self.trainer = CanRotationTrainer(model=model, train_loader=train_loader, val_loader=val_loader,
                                     optimizer=optimizer, scheduler=scheduler, device=cfg.DEVICE)

        # Запуск обучения по эпохам с валидацией и эмиссией сигналов
        for epoch in range(1, self.epochs + 1):
            if self._stop_requested:
                break

            # Тренировка одной эпохи
            train_loss, train_acc = self.trainer.train_epoch(epoch)

            # Проверяем запрос остановки после тренировки
            if getattr(self.trainer, '_stop_requested', False) or self._stop_requested:
                break

            # Валидация
            val_loss, val_acc = self.trainer.validate_epoch(epoch)

            # Шаг scheduler
            try:
                self.trainer.scheduler.step()
            except Exception:
                pass

            # Новая логика сохранения: сохраняем, если ИЛИ улучшилась accuracy, ИЛИ уменьшился val_loss
            improved_acc = val_acc > getattr(self.trainer, 'best_accuracy', 0.0)
            improved_loss = val_loss < getattr(self.trainer, 'best_val_loss', float('inf'))

            if improved_acc:
                self.trainer.best_accuracy = val_acc
            if improved_loss:
                self.trainer.best_val_loss = val_loss

            if improved_acc or improved_loss:
                try:
                    self.trainer.save_checkpoint(epoch, is_best_acc=improved_acc, is_best_loss=improved_loss)
                except Exception:
                    pass

            # Генерируем примеры аугментаций и эмитим сигнал в GUI-поток
            try:
                examples = self._generate_aug_examples(dataset)
                # Emit examples (словарь numpy-arrays with params)
                self.augmentations_signal.emit(examples)
            except Exception:
                pass

            # Эмиссия сигнала с актуальными метриками
            train_loss = getattr(self.trainer, 'last_train_loss', train_loss)
            val_loss = getattr(self.trainer, 'last_val_loss', val_loss)
            train_acc = getattr(self.trainer, 'last_train_acc', train_acc)
            val_acc = getattr(self.trainer, 'last_val_acc', val_acc)
            # Emit epoch, train_loss, val_loss, train_acc, val_acc
            self.epoch_signal.emit(epoch, train_loss, val_loss, train_acc, val_acc)

            # Проверка локального флага на случай stop
            if self._stop_requested:
                break

        # По окончанию вызовем finish
        self.quit()

    def request_stop(self):
        """Запрос остановки: делегируем тренеру и ставим локальный флаг"""
        self._stop_requested = True
        if self.trainer is not None:
            try:
                self.trainer.request_stop()
            except Exception:
                pass
        # Дождёмся завершения потока (в UI вызовется wait())
