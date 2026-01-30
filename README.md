NeuroCan — инструкции по установке и миграции (conda)

Краткое описание
-----------------
Проект NeuroCan использует PyTorch, GUI на PySide6 и ряд библиотек для обработки изображений. Этот документ описывает, как воспроизвести окружение на другом ПК точно в тех же версиях и источниках, которые использовались у автора.

Ключевая информация
-------------------
- Точное зафиксированное pip-список зависимостей записан в `requirements.txt` в корне репозитория.
- Файл `environment_locked.yml` (в корне) содержит conda-экспорт (prefix указывает на Miniconda3 у автора).

Требования к системе
--------------------
- Windows 10/11 x64 (или аналогичная конфигурация с поддержкой CUDA для GPU-версии).
- Рекомендуемая версия Miniconda/conda (важно установить именно эту версию для воспроизводимости):
  - Miniconda3 installer для Windows x86_64 с conda версии 23.11.0
  - Ссылка: https://repo.anaconda.com/miniconda/ (выберите «Miniconda3 Windows 64-bit installer» для conda 23.11.0)

Примечание по версии conda
-------------------------
Точная версия conda, используемая автором окружения, не записана явно в экспортируемых файлах окружения. Я делаю разумное допущение и рекомендую Miniconda3 с conda 23.11.0 — эта версия совместима с pip/pip-tools и conda-каналами, использованными при сборке окружения. Если вы хотите узнать точную локальную версию conda на машине-источнике, выполните на ней:

    conda --version

и сообщите результат — при необходимости я скорректирую инструкции.

Шаги миграции (коротко)
-----------------------
1) Установите Miniconda3 (conda 23.11.0 рекомендовано) и откройте PowerShell.
2) Создайте и активируйте среду (пример имени `sreda`):
   conda create -n sreda python=3.9 -y
   conda activate sreda

3) Установите PyTorch с поддержкой CUDA через conda (если нужна GPU-поддержка).
   Пример (для CUDA 12.1, если у вас драйвер совместим):
   conda install -n sreda -c nvidia -c pytorch pytorch torchvision torchaudio pytorch-cuda=12.1 -y

   Если conda не находит необходимую сборку для вашей GPU-архитектуры/CUDA, используйте pip-колёса PyTorch, как описано на https://pytorch.org/get-started/locally/.

4) После установки PyTorch установите остальные зависимости из `requirements.txt`:
   pip install --no-cache-dir -r requirements.txt

5) Проверьте установку PyTorch и наличие GPU:
   python -c "import torch; print('torch:', torch.__version__); print('torch.version.cuda:', torch.version.cuda); print('cuda_available:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count()); print('device_name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no_gpu')"

6) Запуск GUI (в активированной среде):
   python gui/app.py

Что было зафиксировано и что удалено
-----------------------------------
- Зафиксированные артефакты для переносимости:
  - `requirements.txt` — содержит pip-список пакетов и точные версии (обновлён и соответствует экспортированному окружению).
  - `environment_locked.yml` — conda-экспорт (оставлен без изменений).

- Удалены (как вы просили) лишние дубликаты/старые файлы:
  - `README_INSTALL.md` — удалён из репозитория.
  - `requirements-pinned.txt` — удалён из репозитория.

Рекомендации и отладка
----------------------
- Если при установке PySide6/PySide6-bindings или pyqtgraph возникают ошибки, посмотрите секцию «Qt bindings» в исходном README (старый текст оставлен в истории коммитов). Частые проблемы — конфликты между PyQt6 и PySide6, поэтому удаляйте PyQt6, если он установлен.
- После успешной установки окружения можно зафиксировать его локальное состояние командой:
    conda env export --name sreda --from-history > environment_locked.yml
    pip freeze > requirements-pinned.txt

Если нужно, могу вернуть `requirements-pinned.txt` в репозиторий (сохранён локально у автора) или подготовить zip с колесами/whl для офлайн-установки — скажите, если это необходимо.

Контакты
--------
Если что-то пойдет не так — пришлите выводы команд `conda list`, `pip freeze`, и `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"` — я помогу скорректировать инструкции.
