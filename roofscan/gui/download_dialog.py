"""Diálogo de descarga de imágenes Sentinel-2 desde CDSE.

Flujo:
1. El usuario configura bbox (pre-rellenado desde el mapa), rango de fechas,
   nubosidad máxima y número de resultados.
2. Clic en "Buscar" → ``_SearchWorker`` consulta la API CDSE en background y
   puebla la tabla de escenas disponibles.
3. El usuario selecciona una escena de la tabla.
4. Clic en "Descargar" → ``_DownloadWorker`` descarga la escena .SAFE y la
   convierte a GeoTIFF multibanda via ``safe_to_geotiff``.
5. Al terminar, el diálogo emite ``image_ready(geotiff_path, scl_path)`` y se cierra.

La señal ``image_ready`` es la interfaz con ``MainWindow``, que la usa para
auto-cargar la imagen descargada sin que el usuario tenga que buscar el archivo.
"""

import logging
from pathlib import Path

from PyQt6.QtCore import QDate, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QDateEdit,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workers internos
# ---------------------------------------------------------------------------

class _SearchWorker(QObject):
    """Consulta la API CDSE en background y emite la lista de escenas."""

    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, bbox: tuple, date_range: tuple, max_cloud_pct: int, max_results: int):
        super().__init__()
        self.bbox = bbox
        self.date_range = date_range
        self.max_cloud_pct = max_cloud_pct
        self.max_results = max_results

    def run(self) -> None:
        try:
            from roofscan.core.ingesta.downloader import search_sentinel2
            results = search_sentinel2(
                self.bbox,
                self.date_range,
                self.max_cloud_pct,
                self.max_results,
            )
            self.finished.emit(results)
        except Exception as exc:
            log.error("Error en búsqueda CDSE: %s", exc, exc_info=True)
            self.error.emit(str(exc))


class _DownloadWorker(QObject):
    """Descarga una escena .SAFE y la convierte a GeoTIFF en background."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(str, str)   # (geotiff_path, scl_path)
    error = pyqtSignal(str)

    def __init__(self, scene: dict, output_dir: Path):
        super().__init__()
        self.scene = scene
        self.output_dir = Path(output_dir)

    def run(self) -> None:
        try:
            from roofscan.core.ingesta.downloader import download_by_id
            from roofscan.core.ingesta.safe_loader import safe_to_geotiff

            self.progress.emit(f"Descargando {self.scene['name']}… (puede tardar varios minutos)")
            safe_path = download_by_id(self.scene, self.output_dir)

            self.progress.emit("Convirtiendo .SAFE → GeoTIFF multibanda…")
            geotiff_path = safe_to_geotiff(safe_path, self.output_dir)

            scl_candidate = self.output_dir / (safe_path.stem + "_SCL.tif")
            scl_path = str(scl_candidate) if scl_candidate.exists() else ""

            self.finished.emit(str(geotiff_path), scl_path)

        except Exception as exc:
            log.error("Error en descarga/conversión: %s", exc, exc_info=True)
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Diálogo principal
# ---------------------------------------------------------------------------

class DownloadDialog(QDialog):
    """Diálogo modal para buscar y descargar imágenes Sentinel-2 desde CDSE.

    Signals:
        image_ready (str, str): Emitida cuando la imagen está lista.
            Primer argumento: path al GeoTIFF multibanda.
            Segundo argumento: path al SCL (o cadena vacía si no existe).
    """

    image_ready = pyqtSignal(str, str)

    # Valores por defecto para la zona de prueba (Luján)
    _DEFAULT_BBOX = (-59.15, -34.60, -59.05, -34.53)

    def __init__(self, bbox: tuple | None = None, parent=None):
        super().__init__(parent)
        self._bbox = bbox or self._DEFAULT_BBOX
        self._scenes: list[dict] = []
        self._selected_scene: dict | None = None

        self.setWindowTitle("Descargar imagen Sentinel-2")
        self.setModal(True)
        self.resize(740, 600)

        self._build_ui()
        self._fill_bbox(self._bbox)

    # ------------------------------------------------------------------
    # Construcción de la UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        layout.addWidget(self._build_bbox_group())
        layout.addWidget(self._build_params_group())
        layout.addWidget(self._build_results_group())

        # Status
        self._lbl_status = QLabel("Configurá los parámetros y hacé clic en Buscar escenas.")
        self._lbl_status.setFont(QFont("Segoe UI", 8))
        self._lbl_status.setWordWrap(True)
        layout.addWidget(self._lbl_status)

        # Barra de progreso
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setMaximumHeight(12)
        self._progress.hide()
        layout.addWidget(self._progress)

        # Botones de acción
        btn_row = QHBoxLayout()

        self._btn_search = QPushButton("🔍  Buscar escenas")
        self._btn_search.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self._btn_search.setMinimumHeight(32)
        self._btn_search.clicked.connect(self._on_search)
        btn_row.addWidget(self._btn_search)

        self._btn_download = QPushButton("⬇  Descargar seleccionada")
        self._btn_download.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self._btn_download.setMinimumHeight(32)
        self._btn_download.setEnabled(False)
        self._btn_download.clicked.connect(self._on_download)
        btn_row.addWidget(self._btn_download)

        layout.addLayout(btn_row)

    def _build_bbox_group(self) -> QGroupBox:
        grp = QGroupBox("Área de interés (WGS84)")
        grp.setFont(QFont("Segoe UI", 9))
        lay = QHBoxLayout(grp)
        lay.setSpacing(6)

        self._spin_lon_min = self._coord_spin(lay, "Lon mín:", -180, 180)
        self._spin_lat_min = self._coord_spin(lay, "Lat mín:", -90, 90)
        self._spin_lon_max = self._coord_spin(lay, "Lon máx:", -180, 180)
        self._spin_lat_max = self._coord_spin(lay, "Lat máx:", -90, 90)

        return grp

    def _build_params_group(self) -> QGroupBox:
        grp = QGroupBox("Parámetros de búsqueda")
        grp.setFont(QFont("Segoe UI", 9))
        lay = QHBoxLayout(grp)
        lay.setSpacing(10)

        lay.addWidget(QLabel("Desde:"))
        self._date_start = QDateEdit()
        self._date_start.setCalendarPopup(True)
        self._date_start.setDate(QDate.currentDate().addMonths(-3))
        self._date_start.setFont(QFont("Segoe UI", 9))
        lay.addWidget(self._date_start)

        lay.addWidget(QLabel("Hasta:"))
        self._date_end = QDateEdit()
        self._date_end.setCalendarPopup(True)
        self._date_end.setDate(QDate.currentDate())
        self._date_end.setFont(QFont("Segoe UI", 9))
        lay.addWidget(self._date_end)

        lay.addWidget(QLabel("Nubosidad máx (%):"))
        self._spin_cloud = QSpinBox()
        self._spin_cloud.setRange(0, 100)
        self._spin_cloud.setValue(20)
        self._spin_cloud.setFont(QFont("Segoe UI", 9))
        lay.addWidget(self._spin_cloud)

        lay.addWidget(QLabel("Máx resultados:"))
        self._spin_max = QSpinBox()
        self._spin_max.setRange(1, 20)
        self._spin_max.setValue(10)
        self._spin_max.setFont(QFont("Segoe UI", 9))
        lay.addWidget(self._spin_max)

        return grp

    def _build_results_group(self) -> QGroupBox:
        grp = QGroupBox("Escenas disponibles")
        grp.setFont(QFont("Segoe UI", 9))
        lay = QVBoxLayout(grp)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Nombre", "Fecha", "Nube %", "Tamaño (MB)"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setMinimumHeight(200)
        self._table.itemSelectionChanged.connect(self._on_scene_selected)
        lay.addWidget(self._table)

        return grp

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coord_spin(layout: QHBoxLayout, label: str, lo: float, hi: float) -> QDoubleSpinBox:
        lbl = QLabel(label)
        lbl.setFont(QFont("Segoe UI", 8))
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(5)
        spin.setSingleStep(0.01)
        spin.setFont(QFont("Segoe UI", 8))
        layout.addWidget(lbl)
        layout.addWidget(spin)
        return spin

    def _fill_bbox(self, bbox: tuple) -> None:
        lon_min, lat_min, lon_max, lat_max = bbox
        self._spin_lon_min.setValue(lon_min)
        self._spin_lat_min.setValue(lat_min)
        self._spin_lon_max.setValue(lon_max)
        self._spin_lat_max.setValue(lat_max)

    def _current_bbox(self) -> tuple:
        return (
            self._spin_lon_min.value(),
            self._spin_lat_min.value(),
            self._spin_lon_max.value(),
            self._spin_lat_max.value(),
        )

    def _set_busy(self, busy: bool) -> None:
        self._btn_search.setEnabled(not busy)
        self._btn_download.setEnabled(not busy and self._selected_scene is not None)
        if busy:
            self._progress.show()
        else:
            self._progress.hide()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_search(self) -> None:
        bbox = self._current_bbox()
        date_range = (
            self._date_start.date().toString("yyyy-MM-dd"),
            self._date_end.date().toString("yyyy-MM-dd"),
        )

        self._set_busy(True)
        self._lbl_status.setText("Buscando escenas en CDSE…")
        self._table.setRowCount(0)
        self._selected_scene = None
        self._btn_download.setEnabled(False)

        self._search_thread = QThread()
        self._search_worker = _SearchWorker(
            bbox, date_range,
            self._spin_cloud.value(),
            self._spin_max.value(),
        )
        self._search_worker.moveToThread(self._search_thread)
        self._search_thread.started.connect(self._search_worker.run)
        self._search_worker.finished.connect(self._on_search_done)
        self._search_worker.error.connect(self._on_search_error)
        self._search_worker.finished.connect(self._search_thread.quit)
        self._search_worker.error.connect(self._search_thread.quit)
        self._search_thread.finished.connect(lambda: self._set_busy(False))
        self._search_thread.start()

    def _on_search_done(self, scenes: list) -> None:
        self._table.setRowCount(0)

        if not scenes:
            self._scenes = []
            self._lbl_status.setText(
                "No se encontraron escenas para los parámetros dados. "
                "Probá ampliar el rango de fechas o aumentar la nubosidad máxima."
            )
            return

        # Ordenar por nubosidad ascendente: la mejor escena queda en la primera fila
        self._scenes = sorted(scenes, key=lambda s: s.get("cloud_pct", 0))

        for scene in self._scenes:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(scene.get("name", "")))
            self._table.setItem(row, 1, QTableWidgetItem(scene.get("date", "")))
            self._table.setItem(row, 2, QTableWidgetItem(f"{scene.get('cloud_pct', 0):.1f}"))
            self._table.setItem(row, 3, QTableWidgetItem(f"{scene.get('size_mb', 0):.0f}"))

        # Auto-seleccionar la primera fila (menor nubosidad)
        self._table.selectRow(0)

        best = self._scenes[0]
        self._lbl_status.setText(
            f"Se encontraron {len(self._scenes)} escena(s). "
            f"Pre-seleccionada la de menor nubosidad: {best['date']} ({best['cloud_pct']:.1f}%). "
            "Podés cambiar la selección o hacer clic en Descargar."
        )

    def _on_search_error(self, msg: str) -> None:
        self._lbl_status.setText(f"Error en la búsqueda: {msg}")

    def _on_scene_selected(self) -> None:
        row_idx = self._table.currentRow()
        if 0 <= row_idx < len(self._scenes):
            self._selected_scene = self._scenes[row_idx]
            self._btn_download.setEnabled(True)
        else:
            self._selected_scene = None
            self._btn_download.setEnabled(False)

    def _on_download(self) -> None:
        if not self._selected_scene:
            return

        from roofscan.config import CACHE_DIR

        self._set_busy(True)
        self._lbl_status.setText(
            f"Descargando {self._selected_scene['name']}… "
            "Esto puede tardar varios minutos según el tamaño de la escena."
        )

        self._dl_thread = QThread()
        self._dl_worker = _DownloadWorker(self._selected_scene, CACHE_DIR)
        self._dl_worker.moveToThread(self._dl_thread)
        self._dl_thread.started.connect(self._dl_worker.run)
        self._dl_worker.progress.connect(self._lbl_status.setText)
        self._dl_worker.finished.connect(self._on_download_done)
        self._dl_worker.error.connect(self._on_download_error)
        self._dl_worker.finished.connect(self._dl_thread.quit)
        self._dl_worker.error.connect(self._dl_thread.quit)
        self._dl_thread.finished.connect(lambda: self._set_busy(False))
        self._dl_thread.start()

    def _on_download_done(self, geotiff_path: str, scl_path: str) -> None:
        self._lbl_status.setText(f"Descarga completa: {Path(geotiff_path).name}")
        self.image_ready.emit(geotiff_path, scl_path)
        self.accept()

    def _on_download_error(self, msg: str) -> None:
        self._lbl_status.setText(f"Error: {msg}")
        QMessageBox.warning(self, "Error de descarga", msg)
