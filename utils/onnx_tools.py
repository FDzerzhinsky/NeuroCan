import torch
import os
from pathlib import Path
from typing import Callable, List, Tuple


def list_checkpoints(checkpoint_dir: str) -> List[Path]:
    """Возвращает список .pth файлов в директории checkpoint_dir, отсортированных по времени модификации (новые первыми)."""
    p = Path(checkpoint_dir)
    if not p.exists():
        return []
    files = list(p.glob('*.pth'))
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files


def export_checkpoint_to_onnx(checkpoint_path: str, output_path: str,
                              model_factory: Callable[[], torch.nn.Module],
                              input_shape: Tuple[int,int,int,int] = (1,1,256,536),
                              dynamic_axes: dict = None,
                              opset_version: int = 12):
    """Экспортирует PyTorch checkpoint (.pth) в ONNX.

    - checkpoint_path: путь к .pth файлу (может содержать state_dict или dict с 'model_state_dict')
    - output_path: путь к выходному .onnx
    - model_factory: callable без аргументов, возвращающий инициализированную модель (структуру)
    - input_shape: форма входного тензора (batch, channels, H, W)
    - dynamic_axes: словарь динамических осей для torch.onnx.export
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Создаем модель
    model = model_factory()
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
    else:
        state = checkpoint

    model.load_state_dict(state)
    model.eval()

    # Создаем dummy input
    dummy_input = torch.randn(*input_shape)

    # Устанавливаем динамические оси по-умолчанию, если не переданы
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size'}
        }

    # Экспортируем
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        export_params=True,
        do_constant_folding=True
    )

    return str(output_path)

