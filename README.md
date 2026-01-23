NeuroCan — инструкции по установке (conda)

Краткое описание
-----------------
Проект NeuroCan использует PyTorch, GUI на PySide6 и ряд библиотек для обработки изображений.
На Windows рекомендуется устанавливать PyTorch через conda, особенно при использовании CUDA.

Содержание инструкций
--------------------
1) Создание и активация conda-окружения
2) Удаление старых установок PyTorch (pip/conda)
3) Установка PyTorch (CUDA через conda или pip)
4) Установка остальных зависимостей (pip)
5) Проверки (GPU, PyTorch)
6) Исправление Qt-байндингов и pyqtgraph (если GUI падает)
7) Запуск GUI
8) Ссылки и полезные советы

Шаги (PowerShell — каждая команда в отдельной строке):

# 1. Создать окружение с Python 3.9 (пример, имя среды — `sreda`)
conda create -n sreda python=3.9 -y

# 2. Активировать окружение
conda activate sreda

# 3. Проверить текущие установки PyTorch
conda list | findstr "torch"
# Проверить, установлен ли torch через pip
pip show torch | echo "torch not installed via pip"

# 4. Удаление старых установок (PowerShell-совместимо). Выполняйте только ту ветку, которая применима.
# 4A) Если PyTorch был установлен через pip (часто бывает):
pip uninstall -y torch torchvision torchaudio
# 4B) Если PyTorch был установлен через conda (удаляем conda-пакеты):
conda remove -n sreda --yes pytorch torchvision torchaudio pytorch-cuda cudatoolkit
# В PowerShell НЕ используйте bash-операторы вроде || — выполните нужную команду вручную.

# 5. Установка PyTorch с поддержкой CUDA
# Вы выбрали ставить CUDA 12.1 — используем каналы `pytorch` и `nvidia` и целевой пакет pytorch-cuda=12.1
conda install -n sreda -c nvidia -c pytorch pytorch torchvision torchaudio pytorch-cuda=12.1 -y

# Примечание по версиям CUDA и nvidia-smi:
# - Поле "CUDA Version" в выводе `nvidia-smi` отображает совместимую CUDA Runtime, поддерживаемую драйвером, а не версию CUDA, "установленную" в conda-окружении.
# - PyTorch собирается с наборами архитектур (sm_xx). Если вы видите предупреждение о несовместимости (например, sm_120), это значит, что установленный wheel/conda-пакет PyTorch не содержит скомпилированных ядер для вашей GPU-архитектуры.
# - Для RTX 5070 Ti (sm_120) нужна сборка PyTorch, собранная для CUDA 12.x с поддержкой sm_120; стабильные колёса/пакеты на момент написания чаще доступны для cu121/12.1. Если conda не предоставляет нужный pytorch-cuda=13.1, используйте pytorch-cuda=12.1 и соответствующие pytorch wheels.

# 6. Fallback (pip) — если conda-колонка не находит pytorch-cuda=12.1/13.1
# Пример для cu121 (pip wheels от PyTorch):
# pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# 7. Установка остальных зависимостей (после PyTorch):
# Если ранее стояли PyQt6 — удаляем, чтобы избежать конфликтов биндингов:
pip uninstall -y PyQt6 PyQt6-sip
# Устанавливаем PySide6 и shiboken6 совместимые с Python 3.9:
pip install --no-cache-dir PySide6==6.5.2 shiboken6==6.5.2
# Устанавливаем pyqtgraph (для живых графиков) и tqdm (используется в trainer):
pip install pyqtgraph==0.13.5 tqdm==4.65.0
# Затем устанавливаем остальные зависимости из requirements:
pip install -r requirements.txt

# 8. Проверка установки (выполнить в среде `sreda`):
python -c "import torch; print('torch:', torch.__version__); print('torch.version.cuda:', torch.version.cuda); print('cuda_available:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count()); print('device_name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no_gpu')"

# 9. Запуск GUI (в среде):
python gui/app.py

Примечания и рекомендации
-------------------------
- Частая причина "PackagesNotFoundError" при попытке установить pytorch-cuda — отсутствие нужной метки/канала или попытка удалить/установить пакеты, которые были ранее установлены через pip (в этом случае conda их не видит).
- Если вы полностью перестраиваете окружение и хотите чистую установку, проще удалить среду и создать заново:
    conda deactivate
    conda remove -n sreda --all -y
    conda create -n sreda python=3.9 -y
- На вашей системе `nvidia-smi` показывает CUDA Version: 13.1 и Driver Version: 591.74. Это значит, что драйвер поддерживает CUDA 13.x рантайм; но conda-пакеты pytorch-cuda для win-64 могут быть доступны только для конкретных меток (например, 12.1). Если conda не находит нужный pytorch-cuda, используйте pip-колёса для cu121 или дождитесь появления сборки для cu131.
- После успешной установки зафиксируйте окружение:
    conda env export --name sreda --from-history > environment_locked.yml
    pip freeze > requirements-pinned.txt

Полезные ссылки
---------------
- PyTorch локальная установка (генератор команд): https://pytorch.org/get-started/locally/
- Драйверы NVIDIA: https://www.nvidia.com/Download/index.aspx
- CUDA toolkit (по необходимости): https://developer.nvidia.com/cuda-downloads

Если хотите, могу также подготовить `environment_locked.yml` и `requirements-pinned.txt` после того, как вы подтвердите, что установка прошла успешно — скажите, когда будете готовы.
