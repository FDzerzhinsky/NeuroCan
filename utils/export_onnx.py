import torch
import torch.nn as nn
from models.resnet_model import ResNetCanClassifier
from config.config import cfg


def export_to_onnx():
    # Загружаем модель
    model = ResNetCanClassifier(num_classes=360)

    # Загружаем веса (укажите путь к вашему .pth файлу)
    checkpoint = torch.load('C:\\Users\\user\\PycharmProjects\\NeuroCan\\checkpoints\\best_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Создаем фиктивный вход с правильной размерностью
    # [batch_size, channels, height, width] = [1, 1, 256, 536]
    dummy_input = torch.randn(1, 1, 256, 536)

    # Экспортируем в ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "can_angle_model.onnx",
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=12,
        export_params=True
    )
    print("✅ Модель успешно экспортирована в can_angle_model.onnx")


if __name__ == '__main__':
    export_to_onnx()