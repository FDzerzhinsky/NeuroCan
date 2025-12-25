from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel,
                               QFileDialog, QTabWidget, QFormLayout, QLineEdit, QSpinBox,
                               QMessageBox)
from PySide6.QtCore import Slot
from .training_worker import TrainingWorker
from utils.onnx_tools import list_checkpoints, export_checkpoint_to_onnx
from config.config import cfg

# Попытка импортировать pyqtgraph, при отсутствии — сообщим пользователю в UI
try:
    import pyqtgraph as pg
    _HAS_PYQTGRAPH = True
except Exception:
    pg = None
    _HAS_PYQTGRAPH = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('NeuroCan')
        self.resize(1000, 700)

        self.worker = None

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self._build_training_tab()
        self._build_inference_tab()
        self._build_export_tab()

    def _build_training_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

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
        layout.addLayout(form)
        layout.addWidget(self.device_label)

        # Start/Stop
        self.start_btn = QPushButton('Запустить обучение')
        self.start_btn.clicked.connect(self._start_training)
        self.stop_btn = QPushButton('Остановить обучение')
        self.stop_btn.clicked.connect(self._stop_training)
        self.stop_btn.setEnabled(False)

        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)

        # Plot area (pyqtgraph)
        if _HAS_PYQTGRAPH:
            self.plot_widget = pg.PlotWidget(title='Кривые обучения')
            self.plot_widget.addLegend()
            # Кривые с узловыми точками и русскими подписями
            self.loss_curve = self.plot_widget.plot([], [], pen=pg.mkPen('r', width=2), symbol='o', symbolBrush='r', symbolSize=6, name='Ошибка обучения')
            self.val_curve = self.plot_widget.plot([], [], pen=pg.mkPen('g', width=2), symbol='o', symbolBrush='g', symbolSize=6, name='Ошибка валидации')
            # Настройки оси X: подписи целыми значениями (будет обновляться при каждом epoch)
            self.x_axis = self.plot_widget.getAxis('bottom')
            layout.addWidget(self.plot_widget)
        else:
            layout.addWidget(QLabel('pyqtgraph не установлен — живые графики отключены'))

        tab.setLayout(layout)
        self.tabs.addTab(tab, 'Обучение')

    def _build_inference_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Валидация (Inference) — TODO'))
        tab.setLayout(layout)
        self.tabs.addTab(tab, 'Валидация')

    def _build_export_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.checkpoints_list_label = QLabel('Доступные чекпоинты будут показаны здесь')
        refresh_btn = QPushButton('Обновить')
        refresh_btn.clicked.connect(self._refresh_checkpoints)
        export_btn = QPushButton('Экспортировать выбранный в ONNX')
        export_btn.clicked.connect(self._export_selected)

        layout.addWidget(self.checkpoints_list_label)
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

    def _set_integer_x_ticks(self, x_values):
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
            self.x_axis.setTicks([ticks])
        except Exception:
            # Без фатальных последствий
            pass

    @Slot(int, float, float)
    def _on_epoch(self, epoch, train_loss, val_loss):
        # Обновляем график
        if _HAS_PYQTGRAPH:
            # Текущее содержимое кривых
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

            # Обновляем тики по оси X
            try:
                self._set_integer_x_ticks(x)
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
            reply = QMessageBox.question(self, 'Выход', 'Идёт обучение. Остановить и выйти?', QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
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
        if not files:
            self.checkpoints_list_label.setText('Чекпоинтов не найдено')
        else:
            text = '\n'.join([f.name for f in files[:20]])
            self.checkpoints_list_label.setText(text)

    @Slot()
    def _export_selected(self):
        files = list_checkpoints(str(cfg.CHECKPOINT_DIR))
        if not files:
            QMessageBox.warning(self, 'Экспорт', 'Чекпоинты отсутствуют')
            return
        # For MVP: pick the first one
        cp = files[0]
        out_path, _ = QFileDialog.getSaveFileName(self, 'Сохранить ONNX как', str(cfg.DEFAULT_ONNX_DIR / (cp.stem + '.onnx')), 'ONNX Files (*.onnx)')
        if not out_path:
            return
        try:
            # model factory: import here to avoid circular imports
            from models.resnet_model import ResNetCanClassifier
            export_checkpoint_to_onnx(str(cp), out_path, lambda: ResNetCanClassifier(num_classes=cfg.NUM_CLASSES))
            QMessageBox.information(self, 'Экспорт', f'Экспортирован {out_path}')
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка экспорта', str(e))
