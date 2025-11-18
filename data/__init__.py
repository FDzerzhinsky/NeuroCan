"""
Пакет для работы с данными
"""

from .dataset import SodaCanDataset
from .transforms import get_transforms, TiltAugmentation

__all__ = [
    'SodaCanDataset',
    'get_transforms',
    'TiltAugmentation'
]