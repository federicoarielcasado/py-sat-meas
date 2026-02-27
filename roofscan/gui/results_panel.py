"""Panel de resultados: tabla de techos detectados y resumen de métricas.

Muestra los resultados de la detección en una tabla scrolleable y
un resumen con el área total y cantidad de objetos.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QFrame,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor


class ResultsPanel(QWidget):
    """Panel de resultados de detección.

    Args:
        parent: Widget padre de Qt.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def show_results(self, areas: list[dict], resolution_m: float | None = None) -> None:
        """Popula el panel con los resultados de detección.

        Args:
            areas: Lista de dicts retornada por ``calculate_areas()``.
                   Cada dict tiene ``id``, ``area_m2``, ``area_px``, ``centroid_px``.
            resolution_m: Resolución espacial en m/px para mostrarlo en el resumen.
        """
        self._populate_table(areas)
        self._update_summary(areas, resolution_m)

    def clear(self) -> None:
        """Limpia la tabla y el resumen."""
        self.table.setRowCount(0)
        self.lbl_total_area.setText("—")
        self.lbl_n_objects.setText("—")
        self.lbl_resolution.setText("—")

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- Resumen ---
        summary_box = QGroupBox("Resumen")
        summary_box.setFont(QFont("Segoe UI", 9))
        summary_layout = QVBoxLayout(summary_box)
        summary_layout.setSpacing(4)

        self.lbl_n_objects = self._stat_row(summary_layout, "Techos detectados:", "—")
        self.lbl_total_area = self._stat_row(summary_layout, "Área total cubierta:", "—")
        self.lbl_resolution = self._stat_row(summary_layout, "Resolución espacial:", "—")

        layout.addWidget(summary_box)

        # --- Separador ---
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # --- Tabla de objetos ---
        table_label = QLabel("Objetos detectados")
        table_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        layout.addWidget(table_label)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["ID", "Área (m²)", "Área (px)", "Centroide (fila, col)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setFont(QFont("Segoe UI", 8))
        self.table.setFont(QFont("Segoe UI", 8))
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

    def _stat_row(self, parent_layout, label_text: str, default: str) -> QLabel:
        """Crea una fila de estadística (label + valor) y retorna el QLabel del valor."""
        row = QHBoxLayout()
        row.setSpacing(4)
        lbl = QLabel(label_text)
        lbl.setFont(QFont("Segoe UI", 9))
        lbl.setMinimumWidth(170)
        val = QLabel(default)
        val.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(lbl)
        row.addWidget(val)
        parent_layout.addLayout(row)
        return val

    def _populate_table(self, areas: list[dict]) -> None:
        self.table.setRowCount(0)
        for row_data in areas:
            row = self.table.rowCount()
            self.table.insertRow(row)

            items = [
                str(row_data["id"]),
                f"{row_data['area_m2']:,.1f}",
                str(row_data["area_px"]),
                f"({row_data['centroid_px'][0]:.0f}, {row_data['centroid_px'][1]:.0f})",
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, item)

    def _update_summary(self, areas: list[dict], resolution_m: float | None) -> None:
        total_m2 = sum(r["area_m2"] for r in areas)
        self.lbl_n_objects.setText(str(len(areas)))
        self.lbl_total_area.setText(f"{total_m2:,.1f} m²")
        if resolution_m is not None:
            self.lbl_resolution.setText(f"{resolution_m:.1f} m/px")
        else:
            self.lbl_resolution.setText("—")
