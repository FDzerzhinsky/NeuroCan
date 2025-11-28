import torch
import torch.nn as nn
from models.resnet_model import ResNetCanClassifier
from config.config import cfg


def export_to_onnx():
    # Загружаем модель
    model = ResNetCanClassifier(num_classes=360)

    # Загружаем веса
    checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # ВАЖНО: Используем динамические размеры для высоты и ширины
    # [batch_size, channels, height, width]
    dummy_input = torch.randn(1, 1, 256, 536)  # Можно использовать любой размер

    # Экспортируем в ONNX с ДИНАМИЧЕСКИМИ размерами высоты и ширины
    torch.onnx.export(
        model,
        dummy_input,
        "can_angle_model_dynamic.onnx",
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {
                0: 'batch_size',    # динамический батч
                2: 'height',        # динамическая высота
                3: 'width'          # динамическая ширина
            },
            'output': {
                0: 'batch_size'     # динамический батч
            }
        },
        opset_version=12,
        export_params=True,
        do_constant_folding=True
    )
    print("✅ Модель успешно экспортирована в can_angle_model_dynamic.onnx с динамическими размерами!")


if __name__ == '__main__':
    export_to_onnx()