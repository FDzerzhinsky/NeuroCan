import torch
from pathlib import Path


class Config:
    """Конфигурация проекта (без сайд-эффектов при импорте)."""

    # Пути
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    SNAPS_DIR = DATA_DIR / "snaps"
    LABELS_FILE = SNAPS_DIR / "values.txt"
    # Файл для пользовательских переопределений конфигурации (json)
    USER_CONFIG_FILE = BASE_DIR / 'cfg_overrides.json'

    # Модель
    MODEL_NAME = "resnet18"
    NUM_CLASSES = 360
    INPUT_SIZE = (256, 536)
    # Масштаб предпросмотров в GUI: уменьшение размеров предпросмотра относительно INPUT_SIZE
    # Например, 1/3 — уменьшить обе стороны в 3 раза.
    PREVIEW_SCALE = 1.0 / 3.0

    # Настройки изображений
    GRAYSCALE = True  # Изображения в градациях серого
    INPUT_CHANNELS = 1 if GRAYSCALE else 3

    # Обучение
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Оптимизация / scheduler
    # Возможные значения: 'ReduceLROnPlateau', 'CosineAnnealing', 'OneCycleLR'
    LR_SCHEDULER = 'ReduceLROnPlateau'
    # Параметры для ReduceLROnPlateau
    LR_REDUCE_FACTOR = 0.5
    LR_REDUCE_PATIENCE = 3

    # Параметры для OneCycleLR (если выберете)
    ONE_CYCLE_DIV_FACTOR = 25  # max_lr = LEARNING_RATE, initial_lr = LEARNING_RATE/ONE_CYCLE_DIV_FACTOR

    # Градиентный клиппинг (если <=0 — не применяется)
    GRAD_CLIP_NORM = 1.0

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

    def load_user_config(self, path: str = None) -> bool:
        """Загружает пользовательские переопределения конфигурации из JSON-файла.
        Возвращает True если загрузка прошла и применена, иначе False.
        """
        p = Path(path) if path is not None else self.USER_CONFIG_FILE
        if not p.exists():
            return False
        try:
            import json
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Применяем только те ключи, которые есть в объекте конфигурации
            for k, v in data.items():
                # специальные случаи: кортежи и числа оставляем как есть
                if hasattr(self, k):
                    setattr(self, k, v)
            return True
        except Exception:
            return False

    def save_user_config(self, path: str = None, keys: list = None) -> bool:
        """Сохраняет выбранные поля конфигурации в JSON-файл для последующих запусков.
        По умолчанию сохраняются параметры аугментаций.
        Возвращает True при успехе.
        """
        p = Path(path) if path is not None else self.USER_CONFIG_FILE
        if keys is None:
            keys = [
                'MAX_TILT_ANGLE', 'VERTICAL_SHIFT_PERCENT', 'AUG_P_VERTICAL', 'AUG_P_COLOR', 'AUG_P_NOISE', 'AUG_P_TILT',
                'GAMMA_LIMIT', 'BRIGHTNESS_LIMIT', 'CONTRAST_LIMIT', 'GAUSS_NOISE_VAR', 'MOTION_BLUR_LIMIT', 'MEDIAN_BLUR_LIMIT'
            ]
        data = {}
        for k in keys:
            if hasattr(self, k):
                val = getattr(self, k)
                # Приводим Path к строке при необходимости
                if isinstance(val, Path):
                    val = str(val)
                data[k] = val
        try:
            import json
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False


# Создаем глобальный экземпляр (без побочных эффектов)
cfg = Config()