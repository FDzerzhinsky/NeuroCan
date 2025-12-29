import torch
from pathlib import Path


class Config:
    """Конфигурация проекта (без сайд-эффектов при импорте)."""

    # Пути
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    SNAPS_DIR = DATA_DIR / "snaps"
    LABELS_FILE = SNAPS_DIR / "values.txt"

    # Модель
    MODEL_NAME = "resnet18"
    NUM_CLASSES = 360
    INPUT_SIZE = (256, 536)

    # Настройки изображений
    GRAYSCALE = True  # Изображения в градациях серого
    INPUT_CHANNELS = 1 if GRAYSCALE else 3

    # Обучение
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Аугментация
    MAX_TILT_ANGLE = 3
    # Процент вертикального сдвига (доля от высоты изображения)
    VERTICAL_SHIFT_PERCENT = 0.01
    # Вероятности появления аугментаций (используются в data/transforms)
    AUG_P_VERTICAL = 0.8
    AUG_P_COLOR = 0.5
    AUG_P_NOISE = 0.3
    # Вероятность применения тильта (поворота) в датасете
    AUG_P_TILT = 0.8

    # Дополнительные параметры аугментаций (извлечены из transforms)
    # Gamma limits — в формате (min, max) те же числа, что в оригинале (80..120)
    GAMMA_LIMIT = (80, 120)
    # Brightness/contrast limits (абсолютные значения для RandomBrightnessContrast)
    BRIGHTNESS_LIMIT = 0.1
    CONTRAST_LIMIT = 0.1
    # Gauss noise variance limits — используем как var_limit для A.GaussNoise
    GAUSS_NOISE_VAR = (10.0, 50.0)
    # Размытия: пределы для MotionBlur и MedianBlur
    MOTION_BLUR_LIMIT = 3
    MEDIAN_BLUR_LIMIT = 3

    # Оборудование
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4 if torch.cuda.is_available() else 0

    # Сохранение
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    LOG_DIR = BASE_DIR / "logs"
    DEFAULT_ONNX_DIR = BASE_DIR / "onnx"

    def __init__(self):
        # Никаких mkdir или print в __init__ — явная инициализация через ensure_dirs()
        self._dirs_ensured = False

    def ensure_dirs(self):
        """Создаёт необходимые директории (вызывать явным образом из CLI/GUI)."""
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.DEFAULT_ONNX_DIR.mkdir(parents=True, exist_ok=True)
        self._dirs_ensured = True

    def device_info(self):
        return {
            'device': self.DEVICE,
            'num_workers': self.NUM_WORKERS,
            'grayscale': self.GRAYSCALE,
            'input_channels': self.INPUT_CHANNELS
        }


# Создаем глобальный экземпляр (без побочных эффектов)
cfg = Config()