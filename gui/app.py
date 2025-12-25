import sys
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow
from config.config import cfg


def main():
    # Ensure directories exist
    cfg.ensure_dirs()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

