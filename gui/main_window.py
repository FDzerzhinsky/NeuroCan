from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                               QFileDialog, QTabWidget, QFormLayout, QLineEdit, QSpinBox,
                               QMessageBox, QListWidget, QProgressDialog, QListWidgetItem, QAbstractItemView)
from PySide6.QtCore import Slot, QTimer, Qt, Signal
from .training_worker import TrainingWorker
from utils.onnx_tools import list_checkpoints, export_checkpoint_to_onnx
from config.config import cfg
from pathlib import Path
import threading
from datetime import datetime
import torch


# Попытка импортировать pyqtgraph, при отсутствии — сообщим пользователю в UI
try:
    import pyqtgraph as pg
    _HAS_PYQTGRAPH = True
except Exception:
    pg = None
    _HAS_PYQTGRAPH = False


class MainWindow(QMainWindow):
    export_finished = Signal(bool, str, str)  # success, out_path, err
    def __init__(self):
        super().__init__()
        self.setWindowTitle('NeuroCan')
        self.resize(1200, 700)

        self.worker = None
        self._export_progress = None
        self._export_finalized = False
        # подключаем сигнал, который будет эмититься из фонового потока
        self.export_finished.connect(self._finalize_export)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Попробуем загрузить пользовательские переопределения конфигурации (если есть)
        try:
            loaded = cfg.load_user_config()
            if loaded:
                print('[MainWindow] Loaded user augmentation config overrides from', cfg.USER_CONFIG_FILE)
        except Exception:
            pass

        self._build_training_tab()
        # Вынесем аугментации в отдельную вкладку
        self._build_augmentations_tab()
        self._build_inference_tab()
        self._build_export_tab()

        # Подключаем обновление виджетов аугментаций при переключении вкладок
        try:
            self.tabs.currentChanged.connect(self._on_tab_changed)
        except Exception:
            pass

    def _build_training_tab(self):
        tab = QWidget()
        # Основной вертикальный контейнер: сверху — управление, снизу — графики
        main_v = QVBoxLayout()

        # ВЕРХНИЙ РЯД: поля/кнопки
        top_row = QHBoxLayout()

        # Левая колонка верхнего ряда: форма и кнопки
        left_controls = QVBoxLayout()

        form = QFormLayout()
        self.dataset_path_edit = QLineEdit()
        browse_btn = QPushButton('Обзор...')
        browse_btn.clicked.connect(self._browse_dataset)
        form.addRow('Папка с датасетом:', self.dataset_path_edit)
        form.addRow('', browse_btn)

        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 10000); self.epochs_spin.setValue(cfg.NUM_EPOCHS)
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1, 1024); self.batch_spin.setValue(cfg.BATCH_SIZE)
        self.lr_edit = QLineEdit(str(cfg.LEARNING_RATE))

        form.addRow('Эпохи:', self.epochs_spin)
        form.addRow('Размер батча:', self.batch_spin)
        form.addRow('Скорость обучения (lr):', self.lr_edit)

        # Device info
        device_info = cfg.device_info()
        dev_text = 'GPU доступна' if device_info['device'] == 'cuda' else 'GPU недоступна, обучение на CPU'
        self.device_label = QLabel(f"Устройство: {device_info['device']} ({dev_text})")

        left_controls.addLayout(form)
        left_controls.addWidget(self.device_label)

        # Start/Stop
        self.start_btn = QPushButton('Запустить обучение')
        self.start_btn.clicked.connect(self._start_training)
        self.stop_btn = QPushButton('Остановить обучение')
        self.stop_btn.clicked.connect(self._stop_training)
        self.stop_btn.setEnabled(False)

        # Кнопки в компактном горизонтальном ряду
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        left_controls.addLayout(btn_row)

        # Добавляем левый контрол в верхний ряд (без правой панели аугментаций)
        top_row.addLayout(left_controls, 1)

        # Добавляем верхний ряд в основной вертикальный layout
        main_v.addLayout(top_row)

        # НИЖНИЙ РЯД: графики — занимают всю ширину (левая + правая колонки)
        if _HAS_PYQTGRAPH:
            plots_row = QHBoxLayout()

            # Accuracy plot
            self.acc_plot = pg.PlotWidget(title='Точность')
            self.acc_plot.addLegend()
            self.acc_train_curve = self.acc_plot.plot([], [], pen=pg.mkPen('b', width=2), symbol='o', symbolBrush='b', symbolSize=6, name='Точность обучения')
            self.acc_val_curve = self.acc_plot.plot([], [], pen=pg.mkPen('c', width=2), symbol='o', symbolBrush='c', symbolSize=6, name='Точность валидации')
            self.acc_x_axis = self.acc_plot.getAxis('bottom')
            try:
                self.acc_plot.setYRange(0, 100)
            except Exception:
                pass

            # Loss plot (справа)
            self.plot_widget = pg.PlotWidget(title='Кривые обучения')
            self.plot_widget.addLegend()
            self.loss_curve = self.plot_widget.plot([], [], pen=pg.mkPen('r', width=2), symbol='o', symbolBrush='r', symbolSize=6, name='Ошибка обучения')
            self.val_curve = self.plot_widget.plot([], [], pen=pg.mkPen('g', width=2), symbol='o', symbolBrush='g', symbolSize=6, name='Ошибка валидации')
            self.x_axis = self.plot_widget.getAxis('bottom')

            plots_row.addWidget(self.acc_plot)
            plots_row.addWidget(self.plot_widget)
            # Добавляем графики в основной вертикальный layout
            main_v.addLayout(plots_row)
        else:
            main_v.addWidget(QLabel('pyqtgraph не установлен — живые графики отключены'))

        tab.setLayout(main_v)
        self.tabs.addTab(tab, 'Обучение')

    def _build_augmentations_tab(self):
        # Новая вкладка для параметров аугментации (вынесена из training tab)
        tab = QWidget()
        layout = QVBoxLayout()
        from PySide6.QtWidgets import QGroupBox, QDoubleSpinBox

        # Инициализируем pending-словарь значениями из cfg
        # Попробуем заново загрузить пользовательский файл конфигурации (на случай, если он был создан после старта)
        try:
            cfg.load_user_config()
        except Exception:
            pass

        self._pending_aug = {
            'MAX_TILT_ANGLE': int(getattr(cfg, 'MAX_TILT_ANGLE')),
            'VERTICAL_SHIFT_PERCENT': float(getattr(cfg, 'VERTICAL_SHIFT_PERCENT')),
            'AUG_P_VERTICAL': float(getattr(cfg, 'AUG_P_VERTICAL')),
            'AUG_P_COLOR': float(getattr(cfg, 'AUG_P_COLOR')),
            'AUG_P_NOISE': float(getattr(cfg, 'AUG_P_NOISE')),
            'AUG_P_TILT': float(getattr(cfg, 'AUG_P_TILT')),
            'GAMMA_LIMIT': tuple(getattr(cfg, 'GAMMA_LIMIT')),
            'BRIGHTNESS_LIMIT': float(getattr(cfg, 'BRIGHTNESS_LIMIT')),
            'CONTRAST_LIMIT': float(getattr(cfg, 'CONTRAST_LIMIT')),
            'GAUSS_NOISE_VAR': tuple(getattr(cfg, 'GAUSS_NOISE_VAR')),
            'MOTION_BLUR_LIMIT': int(getattr(cfg, 'MOTION_BLUR_LIMIT')),
            'MEDIAN_BLUR_LIMIT': int(getattr(cfg, 'MEDIAN_BLUR_LIMIT')),
        }

        aug_group = QGroupBox('Параметры аугментации')
        aug_form = QFormLayout()

        # MAX_TILT_ANGLE (int)
        self.max_tilt_spin = QSpinBox()
        self.max_tilt_spin.setRange(0, 45)
        self.max_tilt_spin.setValue(self._pending_aug['MAX_TILT_ANGLE'])
        self.max_tilt_spin.valueChanged.connect(lambda v: self._pending_aug.update({'MAX_TILT_ANGLE': int(v)}))
        aug_form.addRow('Максимальный угол тильта (°):', self.max_tilt_spin)

        # VERTICAL_SHIFT_PERCENT (float)
        self.vert_shift_spin = QDoubleSpinBox()
        self.vert_shift_spin.setDecimals(3)
        self.vert_shift_spin.setRange(0.0, 0.5)
        self.vert_shift_spin.setSingleStep(0.005)
        self.vert_shift_spin.setValue(self._pending_aug['VERTICAL_SHIFT_PERCENT'])
        self.vert_shift_spin.valueChanged.connect(lambda v: self._pending_aug.update({'VERTICAL_SHIFT_PERCENT': float(v)}))
        aug_form.addRow('Вертикальный сдвиг (доля высоты):', self.vert_shift_spin)

        # Вероятности аугментаций
        self.aug_p_vertical_spin = QDoubleSpinBox(); self.aug_p_vertical_spin.setDecimals(3); self.aug_p_vertical_spin.setRange(0.0, 1.0); self.aug_p_vertical_spin.setSingleStep(0.05)
        self.aug_p_vertical_spin.setValue(self._pending_aug['AUG_P_VERTICAL'])
        self.aug_p_vertical_spin.valueChanged.connect(lambda v: self._pending_aug.update({'AUG_P_VERTICAL': float(v)}))
        aug_form.addRow('P вертикальной аугментации:', self.aug_p_vertical_spin)

        self.aug_p_color_spin = QDoubleSpinBox(); self.aug_p_color_spin.setDecimals(3); self.aug_p_color_spin.setRange(0.0, 1.0); self.aug_p_color_spin.setSingleStep(0.05)
        self.aug_p_color_spin.setValue(self._pending_aug['AUG_P_COLOR'])
        self.aug_p_color_spin.valueChanged.connect(lambda v: self._pending_aug.update({'AUG_P_COLOR': float(v)}))
        aug_form.addRow('P цветовых аугментаций:', self.aug_p_color_spin)

        self.aug_p_noise_spin = QDoubleSpinBox(); self.aug_p_noise_spin.setDecimals(3); self.aug_p_noise_spin.setRange(0.0, 1.0); self.aug_p_noise_spin.setSingleStep(0.05)
        self.aug_p_noise_spin.setValue(self._pending_aug['AUG_P_NOISE'])
        self.aug_p_noise_spin.valueChanged.connect(lambda v: self._pending_aug.update({'AUG_P_NOISE': float(v)}))
        aug_form.addRow('P шума/размытий:', self.aug_p_noise_spin)

        # P тильта (поворота) — отложенное обновление
        self.aug_p_tilt_spin = QDoubleSpinBox(); self.aug_p_tilt_spin.setDecimals(3); self.aug_p_tilt_spin.setRange(0.0, 1.0); self.aug_p_tilt_spin.setSingleStep(0.05)
        self.aug_p_tilt_spin.setValue(self._pending_aug['AUG_P_TILT'])
        self.aug_p_tilt_spin.valueChanged.connect(lambda v: self._pending_aug.update({'AUG_P_TILT': float(v)}))
        aug_form.addRow('P тильта (поворота):', self.aug_p_tilt_spin)

        # --- Дополнительные параметры, извлечённые из transforms.py и config.cfg ---
        # Gamma limits (min,max)
        self.gamma_min_spin = QSpinBox(); self.gamma_min_spin.setRange(1, 1000); self.gamma_min_spin.setValue(int(self._pending_aug['GAMMA_LIMIT'][0]))
        self.gamma_max_spin = QSpinBox(); self.gamma_max_spin.setRange(1, 1000); self.gamma_max_spin.setValue(int(self._pending_aug['GAMMA_LIMIT'][1]))
        def _update_gamma_min(v):
            a = int(v)
            b = int(self.gamma_max_spin.value())
            if a > b:
                self.gamma_max_spin.setValue(a)
                b = a
            self._pending_aug['GAMMA_LIMIT'] = (a, b)
        def _update_gamma_max(v):
            a = int(self.gamma_min_spin.value())
            b = int(v)
            if b < a:
                self.gamma_min_spin.setValue(b)
                a = b
            self._pending_aug['GAMMA_LIMIT'] = (a, b)
        self.gamma_min_spin.valueChanged.connect(_update_gamma_min)
        self.gamma_max_spin.valueChanged.connect(_update_gamma_max)
        # Добавим их в одну строку (мини-виджет)
        gamma_row = QWidget()
        gamma_row_l = QHBoxLayout(); gamma_row_l.setContentsMargins(0,0,0,0)
        gamma_row_l.addWidget(self.gamma_min_spin); gamma_row_l.addWidget(QLabel('—')); gamma_row_l.addWidget(self.gamma_max_spin)
        gamma_row.setLayout(gamma_row_l)
        aug_form.addRow('Gamma limits (min—max):', gamma_row)

        # Brightness / Contrast limits
        self.brightness_spin = QDoubleSpinBox(); self.brightness_spin.setDecimals(3); self.brightness_spin.setRange(0.0, 1.0); self.brightness_spin.setSingleStep(0.01); self.brightness_spin.setValue(self._pending_aug['BRIGHTNESS_LIMIT'])
        self.brightness_spin.valueChanged.connect(lambda v: self._pending_aug.update({'BRIGHTNESS_LIMIT': float(v)}))
        aug_form.addRow('Brightness limit:', self.brightness_spin)

        self.contrast_spin = QDoubleSpinBox(); self.contrast_spin.setDecimals(3); self.contrast_spin.setRange(0.0, 1.0); self.contrast_spin.setSingleStep(0.01); self.contrast_spin.setValue(self._pending_aug['CONTRAST_LIMIT'])
        self.contrast_spin.valueChanged.connect(lambda v: self._pending_aug.update({'CONTRAST_LIMIT': float(v)}))
        aug_form.addRow('Contrast limit:', self.contrast_spin)

        # Gauss noise var limits (min,max)
        self.gauss_min_spin = QDoubleSpinBox(); self.gauss_min_spin.setDecimals(1); self.gauss_min_spin.setRange(0.0, 1000.0); self.gauss_min_spin.setValue(float(self._pending_aug['GAUSS_NOISE_VAR'][0]))
        self.gauss_max_spin = QDoubleSpinBox(); self.gauss_max_spin.setDecimals(1); self.gauss_max_spin.setRange(0.0, 1000.0); self.gauss_max_spin.setValue(float(self._pending_aug['GAUSS_NOISE_VAR'][1]))
        def _update_gauss_min(v):
            a = float(v)
            b = float(self.gauss_max_spin.value())
            if a > b:
                self.gauss_max_spin.setValue(a)
                b = a
            self._pending_aug['GAUSS_NOISE_VAR'] = (a, b)
        def _update_gauss_max(v):
            a = float(self.gauss_min_spin.value())
            b = float(v)
            if b < a:
                self.gauss_min_spin.setValue(b)
                a = b
            self._pending_aug['GAUSS_NOISE_VAR'] = (a, b)
        self.gauss_min_spin.valueChanged.connect(_update_gauss_min)
        self.gauss_max_spin.valueChanged.connect(_update_gauss_max)
        gauss_row = QWidget(); gauss_row_l = QHBoxLayout(); gauss_row_l.setContentsMargins(0,0,0,0)
        gauss_row_l.addWidget(self.gauss_min_spin); gauss_row_l.addWidget(QLabel('—')); gauss_row_l.addWidget(self.gauss_max_spin)
        gauss_row.setLayout(gauss_row_l)
        aug_form.addRow('Gauss noise var (min—max):', gauss_row)

        # Motion and Median blur limits
        self.motion_blur_spin = QSpinBox(); self.motion_blur_spin.setRange(1, 31); self.motion_blur_spin.setValue(int(self._pending_aug['MOTION_BLUR_LIMIT']))
        self.motion_blur_spin.valueChanged.connect(lambda v: self._pending_aug.update({'MOTION_BLUR_LIMIT': int(v)}))
        aug_form.addRow('Motion blur limit:', self.motion_blur_spin)

        self.median_blur_spin = QSpinBox(); self.median_blur_spin.setRange(1, 31); self.median_blur_spin.setValue(int(self._pending_aug['MEDIAN_BLUR_LIMIT']))
        self.median_blur_spin.valueChanged.connect(lambda v: self._pending_aug.update({'MEDIAN_BLUR_LIMIT': int(v)}))
        aug_form.addRow('Median blur limit:', self.median_blur_spin)

        aug_group.setLayout(aug_form)
        layout.addWidget(aug_group)

        # После создания виджетов — подгрузим свежие значения из cfg
        self._load_aug_widgets_from_cfg()

        # Добавим кнопку применения изменений
        apply_row = QWidget()
        apply_row_l = QHBoxLayout(); apply_row_l.setContentsMargins(0,0,0,0)
        apply_btn = QPushButton('Принять')
        def _apply_augmentations():
            # Записываем все значения из pending в cfg
            try:
                cfg.MAX_TILT_ANGLE = int(self._pending_aug['MAX_TILT_ANGLE'])
                cfg.VERTICAL_SHIFT_PERCENT = float(self._pending_aug['VERTICAL_SHIFT_PERCENT'])
                cfg.AUG_P_VERTICAL = float(self._pending_aug['AUG_P_VERTICAL'])
                cfg.AUG_P_COLOR = float(self._pending_aug['AUG_P_COLOR'])
                cfg.AUG_P_NOISE = float(self._pending_aug['AUG_P_NOISE'])
                cfg.AUG_P_TILT = float(self._pending_aug['AUG_P_TILT'])
                cfg.GAMMA_LIMIT = tuple(self._pending_aug['GAMMA_LIMIT'])
                cfg.BRIGHTNESS_LIMIT = float(self._pending_aug['BRIGHTNESS_LIMIT'])
                cfg.CONTRAST_LIMIT = float(self._pending_aug['CONTRAST_LIMIT'])
                cfg.GAUSS_NOISE_VAR = tuple(self._pending_aug['GAUSS_NOISE_VAR'])
                cfg.MOTION_BLUR_LIMIT = int(self._pending_aug['MOTION_BLUR_LIMIT'])
                cfg.MEDIAN_BLUR_LIMIT = int(self._pending_aug['MEDIAN_BLUR_LIMIT'])

                # Сохраняем изменения в файл для следующего запуска
                try:
                    ok = cfg.save_user_config()
                    if not ok:
                        QMessageBox.warning(self, 'Аугментации', 'Параметры применены, но не удалось сохранить в файл настроек')
                    else:
                        QMessageBox.information(self, 'Аугментации', 'Параметры аугментаций применены и сохранены в конфиге')
                except Exception:
                    QMessageBox.information(self, 'Аугментации', 'Параметры аугментаций применены (сохранение в файл не удалось)')
            except Exception as e:
                QMessageBox.critical(self, 'Ошибка', f'Не удалось применить параметры: {e}')

        apply_btn.clicked.connect(_apply_augmentations)
        apply_row_l.addStretch(1)
        apply_row_l.addWidget(apply_btn)
        apply_row.setLayout(apply_row_l)
        layout.addWidget(apply_row)

        layout.addStretch(1)
        tab.setLayout(layout)
        self.tabs.addTab(tab, 'Аугментации')

    def _build_inference_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Валидация (Inference) — TODO'))
        tab.setLayout(layout)
        self.tabs.addTab(tab, 'Валидация')

    def _build_export_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Список чекпоинтов (интерактивный)
        self.checkpoint_list = QListWidget()
        # Используем явный enum SelectionMode
        self.checkpoint_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        # Поля для имени выходного файла и кнопки
        controls_row = QHBoxLayout()
        self.onnx_name_edit = QLineEdit()
        self.onnx_name_edit.setPlaceholderText('Введите имя файла .onnx (без расширения)')
        browse_btn = QPushButton('Обзор папки...')
        browse_btn.clicked.connect(self._choose_onnx_folder)
        self.onnx_folder_label = QLabel(str(cfg.DEFAULT_ONNX_DIR))
        controls_row.addWidget(self.onnx_name_edit)
        controls_row.addWidget(browse_btn)
        controls_row.addWidget(self.onnx_folder_label)

        refresh_btn = QPushButton('Обновить')
        refresh_btn.clicked.connect(self._refresh_checkpoints)
        export_btn = QPushButton('Экспортировать выбранный в ONNX')
        export_btn.clicked.connect(self._export_selected)

        layout.addWidget(self.checkpoint_list)
        layout.addLayout(controls_row)
        layout.addWidget(refresh_btn)
        layout.addWidget(export_btn)

        tab.setLayout(layout)
        self.tabs.addTab(tab, 'Экспорт')

    @Slot()
    def _browse_dataset(self):
        path = QFileDialog.getExistingDirectory(self, 'Выберите папку с датасетом')
        if path:
            self.dataset_path_edit.setText(path)

    @Slot()
    def _start_training(self):
        dataset_path = self.dataset_path_edit.text()
        if not dataset_path:
            QMessageBox.warning(self, 'Ошибка', 'Пожалуйста, выберите папку с датасетом')
            return

        epochs = self.epochs_spin.value()
        batch = self.batch_spin.value()
        try:
            lr = float(self.lr_edit.text())
        except ValueError:
            QMessageBox.warning(self, 'Ошибка', 'Неправильное значение скорости обучения')
            return

        # Запускаем worker
        self.worker = TrainingWorker(dataset_path, epochs, batch, lr)
        # Подключаем сигнал с пятью параметрами: epoch, train_loss, val_loss, train_acc, val_acc
        self.worker.epoch_signal.connect(self._on_epoch)
        self.worker.finished.connect(self._on_finished)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker.start()

    @Slot()
    def _stop_training(self):
        if self.worker is None:
            return
        self.worker.request_stop()
        # Ждём завершения потока (небольшой таймаут)
        self.worker.wait(5000)
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)

    def _set_integer_x_ticks(self, x_values, axis):
        """Устанавливает целочисленные тики по оси X на основании текущих эпох."""
        if not _HAS_PYQTGRAPH:
            return
        try:
            xs = sorted(set(int(v) for v in x_values))
            if not xs:
                return
            # Выбираем шаг так, чтобы не было слишком много подписей
            max_ticks = 10
            step = max(1, int(len(xs) / max_ticks))
            ticks = [(i, str(i)) for i in xs if i % step == 0]
            # Гарантируем, что первый и последний присутствуют
            if ticks[0][0] != xs[0]:
                ticks.insert(0, (xs[0], str(xs[0])))
            if ticks[-1][0] != xs[-1]:
                ticks.append((xs[-1], str(xs[-1])))
            axis.setTicks([ticks])
        except Exception:
            # Без фатальных последствий
            pass

    @Slot(int, float, float, float, float)
    def _on_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc):
        # Для отладки выводим значения в консоль
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}%")

        # Обновляем графики
        if _HAS_PYQTGRAPH:
            # Loss
            x = list(self.loss_curve.getData()[0]) if self.loss_curve.getData()[0] is not None else []
            y = list(self.loss_curve.getData()[1]) if self.loss_curve.getData()[1] is not None else []
            x.append(epoch)
            y.append(train_loss)
            self.loss_curve.setData(x, y)

            xv = list(self.val_curve.getData()[0]) if self.val_curve.getData()[0] is not None else []
            yv = list(self.val_curve.getData()[1]) if self.val_curve.getData()[1] is not None else []
            xv.append(epoch)
            yv.append(val_loss)
            self.val_curve.setData(xv, yv)

            # Accuracy
            xa = list(self.acc_train_curve.getData()[0]) if self.acc_train_curve.getData()[0] is not None else []
            ya = list(self.acc_train_curve.getData()[1]) if self.acc_train_curve.getData()[1] is not None else []
            xa.append(epoch)
            ya.append(train_acc)
            self.acc_train_curve.setData(xa, ya)

            xva = list(self.acc_val_curve.getData()[0]) if self.acc_val_curve.getData()[0] is not None else []
            yva = list(self.acc_val_curve.getData()[1]) if self.acc_val_curve.getData()[1] is not None else []
            xva.append(epoch)
            yva.append(val_acc)
            self.acc_val_curve.setData(xva, yva)

            # Обновляем тики по оси X для обоих графиков
            try:
                self._set_integer_x_ticks(x, self.x_axis)
                self._set_integer_x_ticks(xa, self.acc_x_axis)
            except Exception:
                pass

    @Slot()
    def _on_finished(self):
        QMessageBox.information(self, 'Обучение', 'Обучение завершено')
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def closeEvent(self, event):
        # Если идёт обучение, запросим останов и подождём
        if self.worker is not None and self.worker.isRunning():
            reply = QMessageBox.question(self, 'Выход', 'Идёт обучение. Остановить и выйти?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.request_stop()
                # Ждём завершения (блокируя закрытие) — в идеале показать прогресс-диалог
                self.worker.wait(10000)
            else:
                event.ignore()
                return
        event.accept()

    @Slot()
    def _refresh_checkpoints(self):
        files = list_checkpoints(str(cfg.CHECKPOINT_DIR))
        self.checkpoint_list.clear()
        if not files:
            self.checkpoint_list.addItem('Чекпоинтов не найдено')
            return

        # Для каждого файла попытаемся прочитать метаданные чекпоинта (без загрузки больших тензоров)
        for f in files:
            display = f.name
            tooltip = ''
            try:
                ck = torch.load(str(f), map_location='cpu')
                epoch = ck.get('epoch') if isinstance(ck, dict) else None
                best_acc = ck.get('best_accuracy') if isinstance(ck, dict) else None
                best_loss = ck.get('best_val_loss') if isinstance(ck, dict) else None
                epoch_val_acc = ck.get('epoch_val_acc') if isinstance(ck, dict) else None
                epoch_val_loss = ck.get('epoch_val_loss') if isinstance(ck, dict) else None
                parts = []
                if epoch is not None:
                    parts.append(f'ep={epoch}')
                # Показываем приоритетно значения для данной эпохи (epoch_val_*), если их нет — используем best_*
                if epoch_val_acc is not None:
                    parts.append(f'val_acc={epoch_val_acc:.2f}%')
                elif best_acc is not None:
                    parts.append(f'best_acc={best_acc:.2f}%')

                if epoch_val_loss is not None:
                    parts.append(f'val_loss={epoch_val_loss:.4f}')
                elif best_loss is not None:
                    parts.append(f'best_loss={best_loss:.4f}')

                if parts:
                    display = f"{f.name} — {' | '.join(parts)}"
                    tooltip = ', '.join(parts)
            except Exception:
                # Если чтение упало, оставляем только имя файла
                pass
            item = QListWidgetItem(display)
            # Сохраняем оригинальное имя файла в UserRole для корректного доступа при экспорте
            item.setData(Qt.ItemDataRole.UserRole, f.name)
            if tooltip:
                item.setToolTip(tooltip)
            self.checkpoint_list.addItem(item)

        # Предзаполним имя onnx первым элементом
        first_stem = files[0].stem if files else 'model'
        self.onnx_name_edit.setText(first_stem)
        self.onnx_folder_label.setText(str(cfg.DEFAULT_ONNX_DIR))

    @Slot()
    def _export_selected(self):
        # Получаем выбранный чекпоинт
        selected_items = self.checkpoint_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, 'Экспорт', 'Пожалуйста, выберите чекпоинт из списка')
            return
        # Читаем оригинальное имя файла из данных элемента
        data_name = selected_items[0].data(Qt.ItemDataRole.UserRole)
        selected_name = data_name if data_name else selected_items[0].text()
        cp_path = Path(cfg.CHECKPOINT_DIR) / selected_name
        if not cp_path.exists():
            QMessageBox.critical(self, 'Экспорт', f'Файл чекпоинта не найден: {cp_path}')
            return

        # Имя output
        out_name = self.onnx_name_edit.text().strip()
        if not out_name:
            out_name = cp_path.stem

        out_folder = Path(self.onnx_folder_label.text())
        out_folder.mkdir(parents=True, exist_ok=True)
        out_path = str(out_folder / (out_name + '.onnx'))

        # Подтверждение при перезаписи
        if Path(out_path).exists():
            reply = QMessageBox.question(self, 'Перезапись', f'{out_path} уже существует. Перезаписать?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Выполним экспорт в отдельном потоке и покажем индикатор прогресса
        result = {'success': False, 'err': ''}
        # Сбрасываем флаг финализации на случай повторного экспорта
        self._export_finalized = False

        def _do_export():
            try:
                from models.resnet_model import ResNetCanClassifier
                export_checkpoint_to_onnx(str(cp_path), out_path, lambda: ResNetCanClassifier(num_classes=cfg.NUM_CLASSES))
                result['success'] = True
            except Exception as e:
                result['success'] = False
                result['err'] = str(e)

            # Логирование
            try:
                log_file = Path(cfg.LOG_DIR) / 'exports.log'
                with open(log_file, 'a', encoding='utf-8') as lf:
                    ts = datetime.now().isoformat()
                    lf.write(f"{ts}\t{cp_path}\t{out_path}\t{result['success']}\t{result['err'] or ''}\n")
            except Exception:
                pass

            # Лог для отладки
            print(f"[EXPORT THREAD] done, success={result.get('success')}, err={result.get('err')}")

            # Гарантированно планируем финализацию в GUI-потоке
            try:
                # Попробуем также эмитить потоковый сигнал — Qt доставит его в GUI-поток
                try:
                    self.export_finished.emit(result.get('success', False), out_path, result.get('err', ''))
                except Exception:
                    pass
                QTimer.singleShot(0, lambda: self._finalize_export(result.get('success', False), out_path, result.get('err', '')))
            except Exception as e:
                print(f"[EXPORT THREAD] scheduling finalize failed: {e}")

        thread = threading.Thread(target=_do_export, daemon=True)
        # Создаём и показываем неиндетерминированный прогресс
        # Передаём пустую строку вместо None, чтобы соответствовать ожидаемым типам
        self._export_progress = QProgressDialog('Экспорт модели в ONNX...', '', 0, 0, self)
        self._export_progress.setWindowTitle('Экспорт')
        try:
            self._export_progress.setCancelButton(None)
        except Exception:
            pass
        self._export_progress.setModal(True)
        self._export_progress.show()

        # Стартуем фоновый поток; он сам вызовет _finalize_export в основном потоке через QTimer.singleShot
        thread.start()

        # Создаём и запускаем таймер-проверку, чтобы гарантировать финализацию, если сигнал не доставится
        def _poll_thread():
            if not thread.is_alive():
                try:
                    self._export_check_timer.stop()
                except Exception:
                    pass
                # Вызовем финализацию (если ещё не выполнена)
                try:
                    self._finalize_export(result.get('success', False), out_path, result.get('err', ''))
                except Exception:
                    pass

        self._export_check_timer = QTimer(self)
        self._export_check_timer.setInterval(200)
        self._export_check_timer.timeout.connect(_poll_thread)
        self._export_check_timer.start()

    def _finalize_export(self, success: bool, out_path: str, err: str):
        """Финализирует экспорt: закрывает диалог и показывает сообщение (в GUI-потоке)."""
        print(f"[MAIN THREAD] _finalize_export called, success={success}, out_path={out_path}, err={err}")

        # Защита от повторного вызова
        if getattr(self, '_export_finalized', False):
            print('[MAIN THREAD] export already finalized, skipping')
            return
        self._export_finalized = True

        # Остановим таймер-проверку, если она запущена
        try:
            if hasattr(self, '_export_check_timer') and getattr(self, '_export_check_timer') is not None:
                try:
                    self._export_check_timer.stop()
                except Exception:
                    pass
                try:
                    del self._export_check_timer
                except Exception:
                    pass
        except Exception:
            pass

        # Закроем прогресс-диалог (если открыт)
        try:
            if getattr(self, '_export_progress', None) is not None:
                try:
                    self._export_progress.close()
                except Exception:
                    pass
                self._export_progress = None
        except Exception:
            pass

        # Показ результата пользователю
        try:
            if success:
                QMessageBox.information(self, 'Экспорт', f'Экспортирован {out_path}')
            else:
                QMessageBox.critical(self, 'Ошибка экспорта', f"{err}")
        except Exception:
            # ничего критичного — диалог мог быть закрыт/недоступен
            print(f"[MAIN THREAD] show result failed: success={success}, err={err}")

    @Slot()
    def _choose_onnx_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Выберите папку для сохранения ONNX', str(cfg.DEFAULT_ONNX_DIR))
        if folder:
            self.onnx_folder_label.setText(folder)

    def _load_aug_widgets_from_cfg(self):
        """Загружает текущие значения аугментаций из cfg в виджеты и self._pending_aug.
        Метод безопасен при частичной инициализации (проверяет наличие виджетов).
        """
        try:
            # Обновим pending-словарь из cfg (на случай, если cfg был изменён/загружен)
            try:
                self._pending_aug.update({
                    'MAX_TILT_ANGLE': int(getattr(cfg, 'MAX_TILT_ANGLE')),
                    'VERTICAL_SHIFT_PERCENT': float(getattr(cfg, 'VERTICAL_SHIFT_PERCENT')),
                    'AUG_P_VERTICAL': float(getattr(cfg, 'AUG_P_VERTICAL')),
                    'AUG_P_COLOR': float(getattr(cfg, 'AUG_P_COLOR')),
                    'AUG_P_NOISE': float(getattr(cfg, 'AUG_P_NOISE')),
                    'AUG_P_TILT': float(getattr(cfg, 'AUG_P_TILT')),
                    'GAMMA_LIMIT': tuple(getattr(cfg, 'GAMMA_LIMIT')),
                    'BRIGHTNESS_LIMIT': float(getattr(cfg, 'BRIGHTNESS_LIMIT')),
                    'CONTRAST_LIMIT': float(getattr(cfg, 'CONTRAST_LIMIT')),
                    'GAUSS_NOISE_VAR': tuple(getattr(cfg, 'GAUSS_NOISE_VAR')),
                    'MOTION_BLUR_LIMIT': int(getattr(cfg, 'MOTION_BLUR_LIMIT')),
                    'MEDIAN_BLUR_LIMIT': int(getattr(cfg, 'MEDIAN_BLUR_LIMIT')),
                })
            except Exception:
                # если чтение cfg не удалось — оставляем pending как есть
                pass

            # Вспомогательная функция для безопасной установки значения виджета
            def _safe_set(widget_attr, value):
                try:
                    widget = getattr(self, widget_attr, None)
                    if widget is None:
                        return
                    # QSpinBox / QDoubleSpinBox поддерживают setValue
                    widget.setValue(value)
                except Exception:
                    # игнорируем ошибку установки
                    pass

            # Устанавливаем значения в соответствующие виджеты (если они созданы)
            _safe_set('max_tilt_spin', int(self._pending_aug.get('MAX_TILT_ANGLE', 0)))
            _safe_set('vert_shift_spin', float(self._pending_aug.get('VERTICAL_SHIFT_PERCENT', 0.0)))
            _safe_set('aug_p_vertical_spin', float(self._pending_aug.get('AUG_P_VERTICAL', 0.0)))
            _safe_set('aug_p_color_spin', float(self._pending_aug.get('AUG_P_COLOR', 0.0)))
            _safe_set('aug_p_noise_spin', float(self._pending_aug.get('AUG_P_NOISE', 0.0)))
            _safe_set('aug_p_tilt_spin', float(self._pending_aug.get('AUG_P_TILT', 0.0)))

            # Gamma limits (tuple)
            gamma = self._pending_aug.get('GAMMA_LIMIT', (80, 120))
            try:
                _safe_set('gamma_min_spin', int(gamma[0]))
                _safe_set('gamma_max_spin', int(gamma[1]))
            except Exception:
                pass

            # Brightness / Contrast
            _safe_set('brightness_spin', float(self._pending_aug.get('BRIGHTNESS_LIMIT', 0.1)))
            _safe_set('contrast_spin', float(self._pending_aug.get('CONTRAST_LIMIT', 0.1)))

            # Gauss noise limits
            gauss = self._pending_aug.get('GAUSS_NOISE_VAR', (10.0, 50.0))
            try:
                _safe_set('gauss_min_spin', float(gauss[0]))
                _safe_set('gauss_max_spin', float(gauss[1]))
            except Exception:
                pass

            # Motion / Median blur
            _safe_set('motion_blur_spin', int(self._pending_aug.get('MOTION_BLUR_LIMIT', 3)))
            _safe_set('median_blur_spin', int(self._pending_aug.get('MEDIAN_BLUR_LIMIT', 3)))
        except Exception:
            # Без фатального исключения — просто логируем (print) и продолжаем
            try:
                print("Warning: _load_aug_widgets_from_cfg failed")
            except Exception:
                pass

