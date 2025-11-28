import torch
import numpy as np
from pathlib import Path
import sys

# Добавляем путь к проекту для импорта
project_path = Path(__file__).parent
sys.path.append(str(project_path))

from inference import CanRotationPredictor
from config.config import cfg


def verify_onnx_conversion():
    """Проверяет совпадение предсказаний PyTorch и ONNX моделей"""

    # Пути к моделям (убедитесь, что пути правильные)
    pytorch_model_path = "checkpoints/best_model.pth"
    onnx_model_path = "utils/can_angle_model.onnx"

    # Проверяем существование файлов
    if not Path(pytorch_model_path).exists():
        print(f"❌ PyTorch model not found: {pytorch_model_path}")
        return
    if not Path(onnx_model_path).exists():
        print(f"❌ ONNX model not found: {onnx_model_path}")
        return

    print("🔍 Starting ONNX conversion verification...")

    # Создаем предсказатели
    print("📥 Loading PyTorch model...")
    pytorch_predictor = CanRotationPredictor(pytorch_model_path)

    print("📥 Loading ONNX model...")
    onnx_predictor = CanRotationPredictor(onnx_model_path)

    # Создаем тестовый вход (такой же, как при конвертации)
    print("🎲 Generating test input...")
    dummy_input = torch.randn(1, 1, 256, 536)

    # Получаем предсказания от PyTorch модели
    print("🔄 Running PyTorch inference...")
    pytorch_result = pytorch_predictor.predict_from_tensor(dummy_input)

    # Получаем предсказания от ONNX модели
    print("🔄 Running ONNX inference...")
    onnx_result = onnx_predictor.predict_from_tensor(dummy_input)

    # Сравниваем результаты
    print("\n📊 Verification Results:")
    print(f"PyTorch - Angle: {pytorch_result['angle']}°, Confidence: {pytorch_result['confidence']:.4f}")
    print(f"ONNX    - Angle: {onnx_result['angle']}°, Confidence: {onnx_result['confidence']:.4f}")

    # Сравниваем углы
    angle_match = pytorch_result['angle'] == onnx_result['angle']
    confidence_diff = abs(pytorch_result['confidence'] - onnx_result['confidence'])

    print(f"\✅ Angle match: {angle_match}")
    print(f"📈 Confidence difference: {confidence_diff:.6f}")

    # Сравниваем сырые выходы (logits)
    pytorch_output = pytorch_result['raw_output']
    onnx_output = onnx_result['raw_output']

    # Вычисляем различия
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    mean_diff = np.mean(np.abs(pytorch_output - onnx_output))

    print(f"📏 Max output difference: {max_diff:.6f}")
    print(f"📏 Mean output difference: {mean_diff:.6f}")

    # Проверяем корреляцию выходов
    correlation = np.corrcoef(pytorch_output, onnx_output)[0, 1]
    print(f"📊 Output correlation: {correlation:.6f}")

    # Критерии успеха
    success_criteria = [
        angle_match,
        confidence_diff < 0.01,
        max_diff < 0.1,
        correlation > 0.99
    ]

    if all(success_criteria):
        print("\n🎉 ✅ ONNX conversion successful! All checks passed.")
    else:
        print("\n⚠️  ONNX conversion has some differences:")
        if not angle_match:
            print("  - Angles don't match")
        if confidence_diff >= 0.01:
            print(f"  - Confidence difference too large: {confidence_diff:.4f}")
        if max_diff >= 0.1:
            print(f"  - Max output difference too large: {max_diff:.4f}")
        if correlation <= 0.99:
            print(f"  - Output correlation too low: {correlation:.4f}")


def test_with_real_image():
    """Дополнительная проверка с реальным изображением"""
    print("\n🔍 Testing with real image...")

    # Найдите реальное тестовое изображение
    test_images = list(Path("data/snaps").glob("*.jpg")) + list(Path("data/snaps").glob("*.png"))
    if not test_images:
        print("❌ No test images found in data/snaps")
        return

    test_image = test_images[0]
    print(f"📁 Using test image: {test_image}")

    # PyTorch prediction
    pytorch_predictor = CanRotationPredictor("checkpoints/best_model.pth")
    pytorch_result = pytorch_predictor.predict(test_image)

    # ONNX prediction
    onnx_predictor = CanRotationPredictor("can_angle_model.onnx")
    onnx_result = onnx_predictor.predict(test_image)

    print(f"PyTorch: {pytorch_result['angle']}° (conf: {pytorch_result['confidence']:.4f})")
    print(f"ONNX:    {onnx_result['angle']}° (conf: {onnx_result['confidence']:.4f})")
    print(f"Match: {pytorch_result['angle'] == onnx_result['angle']}")


if __name__ == '__main__':
    verify_onnx_conversion()
    test_with_real_image()