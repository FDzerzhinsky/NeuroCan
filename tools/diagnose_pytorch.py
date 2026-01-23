import torch
import importlib

print('Python executable:', __import__('sys').executable)
print('torch:', torch.__version__)
print('torch.version.cuda:', torch.version.cuda)
print('torch.cuda.is_available():', torch.cuda.is_available())
print('torch.cuda.device_count():', torch.cuda.device_count())
if torch.cuda.is_available():
    try:
        print('torch.cuda.get_device_name(0):', torch.cuda.get_device_name(0))
    except Exception as e:
        print('get_device_name error:', e)

try:
    torchvision = importlib.import_module('torchvision')
    print('torchvision:', torchvision.__version__)
except Exception as e:
    print('torchvision import error:', e)

try:
    torchaudio = importlib.import_module('torchaudio')
    print('torchaudio:', torchaudio.__version__)
except Exception as e:
    print('torchaudio import error:', e)

# Print nvidia-smi summary
import subprocess
try:
    out = subprocess.check_output(['nvidia-smi'], text=True)
    print('\n--- nvidia-smi ---')
    for ln in out.splitlines()[:12]:
        print(ln)
except Exception as e:
    print('nvidia-smi failed:', e)
