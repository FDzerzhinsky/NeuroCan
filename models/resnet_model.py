import torch
import torch.nn as nn
import torchvision.models as models
from models.base_model import BaseModel


class ResNetCanClassifier(BaseModel):
    """Модифицированный ResNet для классификации углов банки с поддержкой grayscale"""

    def __init__(self, num_classes=360, backbone='resnet18'):
        super().__init__(num_classes)

        # Импортируем конфиг для получения настроек каналов
        from config.config import cfg

        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            feat_dim = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            feat_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Сохраняем оригинальные веса первого слоя для адаптации
        original_conv1 = self.backbone.conv1
        original_weight = original_conv1.weight.data.clone()

        # Заменяем первый сверточный слой для нашего количества каналов
        self.backbone.conv1 = nn.Conv2d(
            cfg.INPUT_CHANNELS,  # Используем настройку из конфига (1 для grayscale, 3 для RGB)
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )

        # Адаптируем веса для grayscale если нужно
        if cfg.INPUT_CHANNELS == 1 and original_weight.shape[1] == 3:
            # Для grayscale: усредняем веса по RGB каналам
            gray_weight = original_weight.mean(dim=1, keepdim=True)
            self.backbone.conv1.weight.data = gray_weight
            print("✅ Веса первого слоя адаптированы для grayscale")
        elif cfg.INPUT_CHANNELS == original_weight.shape[1]:
            # Если каналы совпадают, используем оригинальные веса
            self.backbone.conv1.weight.data = original_weight
        else:
            # Инициализируем веса заново для других случаев
            nn.init.kaiming_normal_(self.backbone.conv1.weight,
                                    mode='fan_out', nonlinearity='relu')
            print("✅ Веса первого слоя инициализированы заново")

        # Заменяем полносвязный слой
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

        print(f"✅ Модель инициализирована с {cfg.INPUT_CHANNELS} входными каналами")
        print(f"✅ Количество классов: {num_classes}")

    def forward(self, x):
        return self.backbone(x)

    def predict_angle(self, x):
        """Предсказывает угол в градусах"""
        with torch.no_grad():
            logits = self.forward(x)

            # Дополнительная проверка на валидность предсказаний
            if torch.any(logits != logits):  # проверка на NaN
                print("Warning: NaN detected in model output")

            predicted_class = torch.argmax(logits, dim=1)

            # Проверяем, что предсказания в допустимом диапазоне
            if torch.any(predicted_class < 0) or torch.any(predicted_class >= self.num_classes):
                print(f"Warning: Invalid predictions: {predicted_class}")

            return predicted_class.float()