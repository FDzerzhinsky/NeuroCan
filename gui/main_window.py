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

        # Два графика в одной строке: слева Accuracy, справа Loss
        if _HAS_PYQTGRAPH:
            plots_row = QHBoxLayout()

            # Accuracy plot
            self.acc_plot = pg.PlotWidget(title='Точность')
            self.acc_plot.addLegend()
            self.acc_train_curve = self.acc_plot.plot([], [], pen=pg.mkPen('b', width=2), symbol='o', symbolBrush='b', symbolSize=6, name='Точность обучения')
            self.acc_val_curve = self.acc_plot.plot([], [], pen=pg.mkPen('c', width=2), symbol='o', symbolBrush='c', symbolSize=6, name='Точность валидации')
            self.acc_x_axis = self.acc_plot.getAxis('bottom')
            # Ограничим шкалу по Y 0-100 (проценты)
            try:
                self.acc_plot.setYRange(0, 100)
            except Exception:
                pass

            # Loss plot (справа)
            self.plot_widget = pg.PlotWidget(title='Кривые обучения')
            self.plot_widget.addLegend()
            # Кривые с узловыми точками и русскими подписями
            self.loss_curve = self.plot_widget.plot([], [], pen=pg.mkPen('r', width=2), symbol='o', symbolBrush='r', symbolSize=6, name='Ошибка обучения')
            self.val_curve = self.plot_widget.plot([], [], pen=pg.mkPen('g', width=2), symbol='o', symbolBrush='g', symbolSize=6, name='Ошибка валидации')
            self.x_axis = self.plot_widget.getAxis('bottom')

            plots_row.addWidget(self.acc_plot)
            plots_row.addWidget(self.plot_widget)
            layout.addLayout(plots_row)
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
                epoch_val_acc = ck.get('epoch_val_acc') if isinstance(ck, dict) else None
                parts = []
                if epoch is not None:
                    parts.append(f'ep={epoch}')
                if epoch_val_acc is not None:
                    parts.append(f'val={epoch_val_acc:.2f}%')
                elif best_acc is not None:
                    parts.append(f'best={best_acc:.2f}%')
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
