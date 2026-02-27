"""Punto de entrada de la aplicación RoofScan.

Ejecutar con:
    python -m roofscan
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QFont
    from roofscan.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("RoofScan")
    app.setFont(QFont("Segoe UI", 9))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
