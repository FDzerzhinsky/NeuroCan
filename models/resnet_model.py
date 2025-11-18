import torch
import torch.nn as nn
import torchvision.models as models
from .base_model import BaseModel


class ResNetCanClassifier(BaseModel):
    """Модифицированный ResNet для классификации углов банки"""

    def __init__(self, num_classes=360, backbone='resnet18'):
        super().__init__(num_classes)

        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            feat_dim = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            feat_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            3, original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )

        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

        nn.init.kaiming_normal_(self.backbone.conv1.weight,
                                mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.backbone(x)

    def predict_angle(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            predicted_class = torch.argmax(logits, dim=1)
            return predicted_class.float()