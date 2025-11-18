"""
Пакет с моделями нейронных сетей
"""

from .base_model import BaseModel
from .resnet_model import ResNetCanClassifier

__all__ = [
    'BaseModel',
    'ResNetCanClassifier'
]