from PySide6.QtCore import QThread, Signal
from data.dataset import SodaCanDataset
from data.transforms import get_transforms
from training.trainer import CanRotationTrainer
from config.config import cfg
from pathlib import Path


class TrainingWorker(QThread):
    epoch_signal = Signal(int, float, float)  # epoch, train_loss, val_loss

    def __init__(self, dataset_path, epochs, batch_size, lr, parent=None):
        super().__init__(parent)
        self.dataset_path = Path(dataset_path)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self._stop_requested = False
        self.trainer = None

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

            # Сохранение лучшей модели
            if val_acc > self.trainer.best_accuracy:
                self.trainer.best_accuracy = val_acc
                try:
                    self.trainer.save_checkpoint(epoch, is_best=True)
                except Exception:
                    pass

            # Эмиссия сигнала с актуальными метриками
            train_loss = getattr(self.trainer, 'last_train_loss', train_loss)
            val_loss = getattr(self.trainer, 'last_val_loss', val_loss)
            self.epoch_signal.emit(epoch, train_loss, val_loss)

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
