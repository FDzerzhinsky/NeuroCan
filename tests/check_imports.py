import sys

modules = [
    ('numpy', 'import numpy as np; print(np.__version__)'),
    ('scipy', 'import scipy; print(scipy.__version__)'),
    ('pandas', 'import pandas; print(pandas.__version__)'),
    ('matplotlib', 'import matplotlib; print(matplotlib.__version__)'),
    ('cv2', 'import cv2; print(cv2.__version__)'),
    ('albumentations', 'import albumentations; print(albumentations.__version__)'),
    ('sklearn', 'import sklearn; print(sklearn.__version__)'),
    ('skimage', 'import skimage; print(skimage.__version__)'),
    ('yaml', 'import yaml; print(yaml.__version__)'),
    ('PySide6', 'import PySide6; print(PySide6.__version__)'),
    ('pyqtgraph', 'import pyqtgraph; print(pyqtgraph.__version__)'),
    ('onnx', 'import onnx; print(onnx.__version__)'),
    ('onnxruntime', 'import onnxruntime as ort; print(ort.__version__)'),
    ('tqdm', 'import tqdm; print(tqdm.__version__)'),
    ('torch', 'import torch; print(torch.__version__); print("torch.cuda.is_available:", torch.cuda.is_available())'),
]

print('Python', sys.version)
for name, code in modules:
    try:
        exec(code)
        print(f"OK: {name}")
    except Exception as e:
        print(f"ERR: {name} ->", type(e).__name__, e)
