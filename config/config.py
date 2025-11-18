import torch
import os
from pathlib import Path


class Config:
    """Конфигурация проекта"""

    # Пути
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    SNAPS_DIR = DATA_DIR / "snaps"
    LABELS_FILE = DATA_DIR / "values.txt"

    # Модель
    MODEL_NAME = "resnet18"
    NUM_CLASSES = 360
    INPUT_SIZE = (256, 536)

    # Обучение
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Аугментация
    MAX_TILT_ANGLE = 3

    # Оборудование
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4 if torch.cuda.is_available() else 0

    # Сохранение
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    LOG_DIR = BASE_DIR / "logs"

    def __init__(self):
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)


cfg = Config()