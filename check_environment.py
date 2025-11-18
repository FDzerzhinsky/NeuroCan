import torch
import torchvision
import cv2
import albumentations
import numpy as np
import sys
import os


def check_environment():
    print("=" * 50)
    print("ПРОВЕРКА ОКРУЖЕНИЯ NEUROCAN")
    print("=" * 50)

    # Проверка версий
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Albumentations version: {albumentations.__version__}")
    print(f"NumPy version: {np.__version__}")

    # Проверка GPU
    print("\n" + "=" * 30)
    print("ПРОВЕРКА GPU")
    print("=" * 30)

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("❌ CUDA не доступна!")

    # Проверка импортов проекта
    print("\n" + "=" * 30)
    print("ПРОВЕРКА ИМПОРТОВ ПРОЕКТА")
    print("=" * 30)

    try:
        from config.config import cfg
        print("✅ config импортирован")
        print(f"  - Режим: {'Grayscale' if cfg.GRAYSCALE else 'RGB'}")
        print(f"  - Каналы: {cfg.INPUT_CHANNELS}")
    except ImportError as e:
        print(f"❌ Ошибка импорта config: {e}")
        return False

    try:
        from data.dataset import SodaCanDataset
        print("✅ SodaCanDataset импортирован")
    except ImportError as e:
        print(f"❌ Ошибка импорта SodaCanDataset: {e}")
        return False

    try:
        from data.transforms import get_transforms
        print("✅ get_transforms импортирован")
    except ImportError as e:
        print(f"❌ Ошибка импорта get_transforms: {e}")
        return False

    try:
        from models.resnet_model import ResNetCanClassifier
        print("✅ ResNetCanClassifier импортирован")
    except ImportError as e:
        print(f"❌ Ошибка импорта ResNetCanClassifier: {e}")
        return False

    try:
        from training.trainer import CanRotationTrainer
        print("✅ CanRotationTrainer импортирован")
    except ImportError as e:
        print(f"❌ Ошибка импорта CanRotationTrainer: {e}")
        return False

    print("\n✅ Все модули проекта импортируются успешно!")

    # Тест работы с тензорами на GPU
    print("\n" + "=" * 30)
    print("ТЕСТ РАБОТЫ С GPU")
    print("=" * 30)

    if cuda_available:
        try:
            # Создаем тестовый тензор с правильным количеством каналов
            from config.config import cfg
            channels = cfg.INPUT_CHANNELS
            x = torch.randn(channels, 256, 536).cuda()
            y = torch.randn(channels, 256, 536).cuda()
            z = x + y

            print(f"✅ Тестовый тензор на GPU: {z.shape} (каналы: {channels})")
            print(f"✅ Устройство тензора: {z.device}")

            # Тест производительности
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(100):
                _ = x * y
            end.record()

            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end)

            print(f"✅ Время выполнения 100 операций: {elapsed_time:.2f} ms")

            # Проверка памяти
            memory_allocated = torch.cuda.memory_allocated() / 1024 ** 2
            print(f"✅ Память GPU использовано: {memory_allocated:.1f} MB")

        except Exception as e:
            print(f"❌ Ошибка работы с GPU: {e}")
            return False
    else:
        print("⚠️  Пропускаем тесты GPU - CUDA не доступна")

    print("\n" + "=" * 50)
    print("ПРОВЕРКА ЗАВЕРШЕНА")
    print("=" * 50)

    return True


if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)