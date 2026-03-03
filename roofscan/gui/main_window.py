"""Ventana principal de RoofScan.

GUI con:
- Panel de localización: búsqueda por dirección, overlay de parcelas CartoARBA, selección por clic
- Panel de controles: carga de imagen, parámetros de detección, exportación
- Vista de imagen: MapWidget con overlay de detección y parcelas
- Panel de resultados: tabla de techos y resumen de área

Ejecutar con: python -m roofscan
"""

import logging
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QPushButton, QLabel, QFileDialog,
    QGroupBox, QDoubleSpinBox, QSpinBox, QComboBox,
    QMessageBox, QTabWidget, QCheckBox,
    QProgressBar, QSizePolicy, QLineEdit, QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont

from roofscan.gui.map_widget import MapWidget
from roofscan.gui.results_panel import ResultsPanel
from roofscan.gui.validation_panel import ValidationPanel

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workers (QThread)
# ---------------------------------------------------------------------------

class AnalysisWorker(QObject):
    """Ejecuta el pipeline completo de detección en un QThread separado.

    Soporta dos motores intercambiables:
    - "clasico": índices espectrales (NDVI/NDBI/NDWI) + morfología OpenCV
    - "unet": U-Net con pesos guardados en data/models/unet_best.pth
    """
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, filepath: str, params: dict, scl_filepath: str | None = None):
        super().__init__()
        self.filepath = filepath
        self.params = params
        self.scl_filepath = scl_filepath

    def run(self) -> None:
        try:
            from roofscan.core.ingesta.loader import load_geotiff
            from roofscan.core.preproceso.pipeline import run_preprocessing, PreprocessConfig
            from roofscan.core.deteccion.clasico.morphology import run_morphology, MorphologyConfig
            from roofscan.core.calculo.area_calculator import calculate_areas

            engine = self.params.get("engine", "clasico")

            self.progress.emit("Cargando imagen…")
            data = load_geotiff(self.filepath)

            # Cargar SCL si se proveyó
            scl_array = None
            if self.scl_filepath:
                self.progress.emit("Cargando SCL (máscara de nubes)…")
                scl_data = load_geotiff(self.scl_filepath)
                scl_array = scl_data["array"][0]   # Single-band → (H, W)

            self.progress.emit("Preprocesando (reproyección + normalización)…")
            processed = run_preprocessing(
                data,
                scl_array=scl_array,
                config=PreprocessConfig(s2_scale_factor=self.params.get("scale_factor", 10_000.0)),
            )

            arr = processed["array"]
            if engine == "unet":
                detection = self._run_unet(arr)
            else:
                detection = self._run_clasico(arr)

            self.progress.emit("Limpieza morfológica…")
            morph = run_morphology(
                detection["mask"],
                config=MorphologyConfig(min_area_px=self.params["min_area_px"]),
            )

            self.progress.emit("Calculando áreas…")
            resolution_m = processed.get("resolution_m") or 10.0
            areas = calculate_areas(morph["labels"], resolution_m=resolution_m)

            self.finished.emit({
                "data": processed,
                "detection": detection,
                "morph": morph,
                "areas": areas,
                "resolution_m": resolution_m,
                "engine": engine,
            })
        except Exception as exc:
            log.error("Error en análisis: %s", exc, exc_info=True)
            self.error.emit(str(exc))

    def _run_clasico(self, arr: np.ndarray) -> dict:
        """Motor clásico: índices espectrales NDVI/NDBI/NDWI."""
        from roofscan.core.deteccion.clasico.spectral_indices import detect_roofs, DetectionConfig
        self.progress.emit("Calculando índices espectrales (motor clásico)…")
        return detect_roofs(
            arr,
            config=DetectionConfig(
                ndvi_max=self.params["ndvi_max"],
                ndbi_min=self.params["ndbi_min"],
                ndwi_max=self.params["ndwi_max"],
            ),
        )

    def _run_unet(self, arr: np.ndarray) -> dict:
        """Motor U-Net: inferencia con modelo entrenado."""
        from roofscan.config import MODELS_DIR
        from roofscan.core.deteccion.dl.unet import build_unet, load_weights, get_device
        from roofscan.core.deteccion.dl.predictor import predict_mask

        weights_path = MODELS_DIR / "unet_best.pth"
        if not weights_path.exists():
            raise FileNotFoundError(
                "No se encontraron pesos entrenados para U-Net.\n\n"
                "Para usar el motor U-Net:\n"
                "  1. Ejecutá al menos un análisis con el motor Clásico.\n"
                "  2. Guardá la detección como feedback (pestaña Validación).\n"
                "  3. Reiterá con al menos 2 imágenes y reentrená el modelo."
            )

        self.progress.emit("Cargando modelo U-Net…")
        model = build_unet(pretrained=False)
        model = load_weights(model, weights_path)
        device = get_device()

        self.progress.emit(f"Ejecutando inferencia U-Net ({device.upper()})…")
        raw_mask = predict_mask(model, arr, device=device)

        return {
            "mask": raw_mask,
            "ndvi": None,
            "ndbi": None,
            "ndwi": None,
            "coverage_pct": float(raw_mask.sum()) / raw_mask.size * 100.0,
            "detection_config": {"engine": "unet"},
        }


class LocationWorker(QObject):
    """Geocodifica y descarga el overlay WMS en background."""
    finished = pyqtSignal(float, float, object, tuple)  # lat, lon, wms_array, bbox
    error = pyqtSignal(str)

    def __init__(self, address: str, radius_km: float = 0.5):
        super().__init__()
        self.address = address
        self.radius_km = radius_km

    def run(self) -> None:
        try:
            from roofscan.core.ingesta.carto_arba import (
                geocode_address, get_parcelas_image, bbox_from_latlon
            )
            lat, lon = geocode_address(self.address)
            bbox = bbox_from_latlon(lat, lon, self.radius_km)
            wms_array, _ = get_parcelas_image(bbox, width=512, height=512)
            self.finished.emit(lat, lon, wms_array, bbox)
        except Exception as exc:
            log.error("Error de localización: %s", exc, exc_info=True)
            self.error.emit(str(exc))


class RetrainWorker(QObject):
    """Ejecuta el fine-tuning del modelo U-Net en un QThread separado."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, epochs: int = 20, batch_size: int = 4):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size

    def run(self) -> None:
        try:
            from roofscan.config import FEEDBACK_DIR, MODELS_DIR
            from roofscan.core.deteccion.dl.unet import build_unet, load_weights
            from roofscan.core.deteccion.dl.trainer import fine_tune

            self.progress.emit("Construyendo modelo U-Net…")
            model = build_unet(pretrained=False)

            # Cargar pesos existentes si los hay (fine-tune sobre modelo previo)
            weights_path = MODELS_DIR / "unet_best.pth"
            if weights_path.exists():
                try:
                    model = load_weights(model, weights_path, strict=True)
                    self.progress.emit("Pesos anteriores cargados. Continuando fine-tune…")
                except Exception:
                    self.progress.emit("Iniciando fine-tune desde cero (pesos ImageNet)…")

            self.progress.emit(f"Entrenando {self.epochs} épocas…")
            history = fine_tune(
                model,
                dataset_dir=FEEDBACK_DIR,
                epochs=self.epochs,
                batch_size=self.batch_size,
            )
            self.finished.emit(history)
        except Exception as exc:
            log.error("Error en reentrenamiento: %s", exc, exc_info=True)
            self.error.emit(str(exc))


class ParcelInfoWorker(QObject):
    """Consulta GetFeatureInfo de una parcela en background."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, bbox, px, py, map_w, map_h):
        super().__init__()
        self.bbox = bbox
        self.px = px
        self.py = py
        self.map_w = map_w
        self.map_h = map_h

    def run(self) -> None:
        try:
            from roofscan.core.ingesta.carto_arba import get_parcel_info
            info = get_parcel_info(self.bbox, self.px, self.py, self.map_w, self.map_h)
            self.finished.emit(info)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Ventana principal
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Ventana principal de RoofScan."""

    APP_TITLE = "RoofScan — Detección de Superficies Cubiertas"

    def __init__(self):
        super().__init__()
        self._results = None
        self._current_file = None
        self._scl_file = None                # ruta al archivo SCL (cloud mask), opcional
        self._current_bbox_wgs84 = None     # bbox WGS84 del mapa visible
        self._selected_parcel = None         # info de la parcela seleccionada
        self._worker_thread = None

        self.setWindowTitle(self.APP_TITLE)
        self.resize(1350, 820)
        self.setMinimumSize(960, 620)

        self._build_ui()
        self._connect_signals()
        self.statusBar().showMessage("Listo. Buscá una dirección o cargá una imagen GeoTIFF.")

    # ------------------------------------------------------------------
    # Construcción de la UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(6, 6, 6, 4)
        root_layout.setSpacing(4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Panel izquierdo
        left = QWidget()
        left.setMaximumWidth(320)
        left.setMinimumWidth(260)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(8)

        left_layout.addWidget(self._build_location_panel())
        left_layout.addWidget(self._build_file_panel())
        left_layout.addWidget(self._build_detection_panel())
        left_layout.addWidget(self._build_view_panel())

        self.btn_run = QPushButton("▶  Detectar techos")
        self.btn_run.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.btn_run.setEnabled(False)
        self.btn_run.setMinimumHeight(36)
        left_layout.addWidget(self.btn_run)

        left_layout.addWidget(self._build_export_panel())
        left_layout.addStretch()
        splitter.addWidget(left)

        # Panel central: tabs con mapa, resultados y validación
        tabs = QTabWidget()
        tabs.setFont(QFont("Segoe UI", 9))
        self.map_widget = MapWidget()
        tabs.addTab(self.map_widget, "Imagen / Detección")
        self.results_panel = ResultsPanel()
        tabs.addTab(self.results_panel, "Resultados")
        self.validation_panel = ValidationPanel()
        tabs.addTab(self.validation_panel, "Validación")
        splitter.addWidget(tabs)

        splitter.setSizes([310, 1040])
        root_layout.addWidget(splitter)

        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setMaximumHeight(14)
        self.progress_bar.hide()
        root_layout.addWidget(self.progress_bar)

    def _build_location_panel(self) -> QGroupBox:
        box = QGroupBox("Localización")
        box.setFont(QFont("Segoe UI", 9))
        layout = QVBoxLayout(box)
        layout.setSpacing(5)

        # Búsqueda por dirección
        self.txt_address = QLineEdit()
        self.txt_address.setPlaceholderText("Ej: Av. San Martín 1234, Luján")
        self.txt_address.setFont(QFont("Segoe UI", 9))
        self.txt_address.setToolTip(
            "Ingresá una dirección o lugar. "
            "Se geocodifica con OpenStreetMap/Nominatim."
        )
        layout.addWidget(self.txt_address)

        row1 = QHBoxLayout()
        self.btn_search = QPushButton("🔍 Buscar")
        self.btn_search.setFont(QFont("Segoe UI", 9))
        row1.addWidget(self.btn_search)
        layout.addLayout(row1)

        # Toggle overlay de parcelas ARBA
        self.chk_arba = QCheckBox("Mostrar parcelas ARBA")
        self.chk_arba.setFont(QFont("Segoe UI", 9))
        self.chk_arba.setEnabled(False)
        self.chk_arba.setToolTip(
            "Superpone la capa catastral de CartoARBA (WMS IDERA).\n"
            "Requiere haber buscado una dirección primero."
        )
        layout.addWidget(self.chk_arba)

        # Info de parcela seleccionada
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        lbl_parcela = QLabel("Parcela seleccionada:")
        lbl_parcela.setFont(QFont("Segoe UI", 8))
        layout.addWidget(lbl_parcela)

        self.lbl_parcel_info = QLabel("— (hacé clic sobre el mapa)")
        self.lbl_parcel_info.setFont(QFont("Segoe UI", 8))
        self.lbl_parcel_info.setWordWrap(True)
        self.lbl_parcel_info.setStyleSheet("color: #495057;")
        layout.addWidget(self.lbl_parcel_info)

        self.btn_use_parcel = QPushButton("Usar esta parcela como área de análisis")
        self.btn_use_parcel.setFont(QFont("Segoe UI", 8))
        self.btn_use_parcel.setEnabled(False)
        self.btn_use_parcel.setToolTip(
            "Descarga Sentinel-2 para el bbox de la parcela seleccionada "
            "y la usa como zona de análisis."
        )
        layout.addWidget(self.btn_use_parcel)

        return box

    def _build_file_panel(self) -> QGroupBox:
        box = QGroupBox("Imagen satelital")
        box.setFont(QFont("Segoe UI", 9))
        layout = QVBoxLayout(box)

        self.lbl_file = QLabel("Sin archivo cargado")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setFont(QFont("Segoe UI", 8))
        self.lbl_file.setStyleSheet("color: #6c757d;")
        layout.addWidget(self.lbl_file)

        row_open = QHBoxLayout()
        self.btn_open = QPushButton("Abrir GeoTIFF…")
        self.btn_open.setToolTip("Cargar imagen satelital en formato GeoTIFF")
        row_open.addWidget(self.btn_open)
        layout.addLayout(row_open)

        self.btn_download_s2 = QPushButton("⬇  Descargar Sentinel-2…")
        self.btn_download_s2.setFont(QFont("Segoe UI", 8))
        self.btn_download_s2.setToolTip(
            "Buscar y descargar una imagen Sentinel-2 L2A desde CDSE.\n"
            "Requiere credenciales CDSE configuradas en el archivo .env."
        )
        layout.addWidget(self.btn_download_s2)

        self.btn_draw_bbox = QPushButton("⬚  Dibujar área de análisis")
        self.btn_draw_bbox.setFont(QFont("Segoe UI", 8))
        self.btn_draw_bbox.setCheckable(True)
        self.btn_draw_bbox.setToolTip(
            "Arrastrá el ratón sobre el mapa para definir el área de descarga.\n"
            "El bbox dibujado se usará como zona de búsqueda en Descargar Sentinel-2."
        )
        layout.addWidget(self.btn_draw_bbox)

        # SCL — máscara de nubes (opcional, Sentinel-2 L2A)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        lbl_scl_title = QLabel("SCL (máscara de nubes, opcional):")
        lbl_scl_title.setFont(QFont("Segoe UI", 8))
        layout.addWidget(lbl_scl_title)

        self.lbl_scl = QLabel("Sin SCL cargado")
        self.lbl_scl.setWordWrap(True)
        self.lbl_scl.setFont(QFont("Segoe UI", 8))
        self.lbl_scl.setStyleSheet("color: #6c757d;")
        layout.addWidget(self.lbl_scl)

        row_scl = QHBoxLayout()
        self.btn_open_scl = QPushButton("Cargar SCL…")
        self.btn_open_scl.setFont(QFont("Segoe UI", 8))
        self.btn_open_scl.setToolTip(
            "Archivo SCL (Scene Classification Layer) de Sentinel-2 L2A.\n"
            "Permite enmascarar nubes y sombras antes de la detección.\n"
            "Típicamente: *_SCL_20m.tif o *_SCL.tif en la carpeta de la escena."
        )
        self.btn_clear_scl = QPushButton("✕")
        self.btn_clear_scl.setFixedWidth(28)
        self.btn_clear_scl.setEnabled(False)
        self.btn_clear_scl.setToolTip("Quitar SCL cargado")
        row_scl.addWidget(self.btn_open_scl)
        row_scl.addWidget(self.btn_clear_scl)
        layout.addLayout(row_scl)

        return box

    def _build_detection_panel(self) -> QGroupBox:
        box = QGroupBox("Parámetros de detección")
        box.setFont(QFont("Segoe UI", 9))
        layout = QVBoxLayout(box)
        layout.setSpacing(6)

        # Selector de motor
        lbl_eng = QLabel("Motor:")
        lbl_eng.setFont(QFont("Segoe UI", 8))
        layout.addWidget(lbl_eng)
        self.combo_engine = QComboBox()
        self.combo_engine.addItems([
            "Clásico (índices espectrales)",
            "U-Net (Deep Learning)",
        ])
        self.combo_engine.setFont(QFont("Segoe UI", 9))
        self.combo_engine.setToolTip(
            "Clásico: usa NDVI/NDBI/NDWI, sin GPU, siempre disponible.\n"
            "U-Net: red neuronal, requiere pesos en data/models/unet_best.pth\n"
            "(reentrenar desde la pestaña Validación)."
        )
        self.combo_engine.currentIndexChanged.connect(self._on_engine_changed)
        layout.addWidget(self.combo_engine)

        # Separador
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # Parámetros motor clásico
        self.spin_ndvi = self._labeled_spin(layout, "NDVI máx (vegetación):", -1.0, 1.0, 0.20, 0.05,
            "Píxeles con NDVI > umbral se descartan como vegetación")
        self.spin_ndbi = self._labeled_spin(layout, "NDBI mín (construido):", -1.0, 1.0, 0.10, 0.05,
            "Píxeles con NDBI < umbral se descartan como no construidos")
        self.spin_ndwi = self._labeled_spin(layout, "NDWI máx (agua):", -1.0, 1.0, 0.05, 0.05,
            "Píxeles con NDWI > umbral se descartan como agua")
        self.spin_min_area = self._labeled_int_spin(layout, "Área mín (px):", 1, 500, 5,
            "Objetos con menos píxeles se eliminan")
        return box

    def _build_view_panel(self) -> QGroupBox:
        box = QGroupBox("Visualización")
        box.setFont(QFont("Segoe UI", 9))
        layout = QVBoxLayout(box)
        self.combo_view = QComboBox()
        self.combo_view.addItems(["RGB True Color", "Overlay detección", "NDVI", "NDBI", "NDWI"])
        self.combo_view.setFont(QFont("Segoe UI", 9))
        layout.addWidget(self.combo_view)
        return box

    def _build_export_panel(self) -> QGroupBox:
        box = QGroupBox("Exportar resultados")
        box.setFont(QFont("Segoe UI", 9))
        layout = QVBoxLayout(box)

        self.btn_export_geotiff = QPushButton("Guardar GeoTIFF procesado")
        self.btn_export_geotiff.setEnabled(False)
        layout.addWidget(self.btn_export_geotiff)

        self.btn_export_png = QPushButton("Guardar PNG de previsualización")
        self.btn_export_png.setEnabled(False)
        layout.addWidget(self.btn_export_png)

        self.btn_export_geojson = QPushButton("Guardar GeoJSON de techos")
        self.btn_export_geojson.setEnabled(False)
        layout.addWidget(self.btn_export_geojson)

        self.btn_export_shp = QPushButton("Guardar Shapefile de techos")
        self.btn_export_shp.setEnabled(False)
        layout.addWidget(self.btn_export_shp)

        self.btn_export_csv = QPushButton("Guardar CSV de áreas")
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.setToolTip(
            "Exporta un CSV con id, área (m²), área (px) y centroide\n"
            "por cada objeto detectado."
        )
        layout.addWidget(self.btn_export_csv)

        return box

    # ------------------------------------------------------------------
    # Señales
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self.btn_search.clicked.connect(self._on_search_address)
        self.txt_address.returnPressed.connect(self._on_search_address)
        self.chk_arba.stateChanged.connect(self._on_toggle_arba_overlay)
        self.btn_use_parcel.clicked.connect(self._on_use_parcel)
        self.btn_open.clicked.connect(self._on_open_file)
        self.btn_download_s2.clicked.connect(self._on_download_image)
        self.btn_open_scl.clicked.connect(self._on_open_scl_file)
        self.btn_clear_scl.clicked.connect(self._on_clear_scl)
        self.btn_run.clicked.connect(self._on_run_detection)
        self.combo_view.currentIndexChanged.connect(self._on_view_changed)
        self.btn_export_geotiff.clicked.connect(self._on_export_geotiff)
        self.btn_export_png.clicked.connect(self._on_export_png)
        self.btn_export_geojson.clicked.connect(self._on_export_geojson)
        self.btn_export_shp.clicked.connect(self._on_export_shp)
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        self.btn_draw_bbox.toggled.connect(self._on_toggle_draw_bbox)
        self.map_widget.geo_clicked.connect(self._on_map_geo_click)
        self.map_widget.bbox_selected.connect(self._on_bbox_drawn)

    # ------------------------------------------------------------------
    # Slots — Localización y CartoARBA
    # ------------------------------------------------------------------

    def _on_engine_changed(self, idx: int) -> None:
        """Habilita/deshabilita parámetros según el motor seleccionado."""
        is_clasico = (idx == 0)
        for spin in (self.spin_ndvi, self.spin_ndbi, self.spin_ndwi):
            spin.setEnabled(is_clasico)

    def _on_open_scl_file(self) -> None:
        """Carga el archivo SCL (Scene Classification Layer) para cloud masking."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Cargar SCL (máscara de nubes)", "",
            "Archivos raster (*.tif *.tiff *.img);;Todos los archivos (*)"
        )
        if not path:
            return
        self._scl_file = path
        self.lbl_scl.setText(Path(path).name)
        self.lbl_scl.setStyleSheet("color: #155724;")
        self.btn_clear_scl.setEnabled(True)
        self.statusBar().showMessage(f"SCL cargado: {Path(path).name}")

    def _on_clear_scl(self) -> None:
        """Quita el archivo SCL."""
        self._scl_file = None
        self.lbl_scl.setText("Sin SCL cargado")
        self.lbl_scl.setStyleSheet("color: #6c757d;")
        self.btn_clear_scl.setEnabled(False)
        self.statusBar().showMessage("SCL eliminado. El análisis no aplicará cloud masking.")

    def _on_search_address(self) -> None:
        address = self.txt_address.text().strip()
        if not address:
            return

        self._set_busy(True, f"Buscando: {address}…")
        self.btn_search.setEnabled(False)

        self._loc_thread = QThread()
        self._loc_worker = LocationWorker(address)
        self._loc_worker.moveToThread(self._loc_thread)
        self._loc_thread.started.connect(self._loc_worker.run)
        self._loc_worker.finished.connect(self._on_location_found)
        self._loc_worker.error.connect(self._on_location_error)
        self._loc_worker.finished.connect(self._loc_thread.quit)
        self._loc_worker.error.connect(self._loc_thread.quit)
        self._loc_thread.finished.connect(lambda: (
            self.btn_search.setEnabled(True),
            self._set_busy(False, ""),
        ))
        self._loc_thread.start()

    def _on_location_found(self, lat: float, lon: float, wms_array, bbox: tuple) -> None:
        self._current_bbox_wgs84 = bbox
        self._wms_array_cache = wms_array

        # Mostrar mapa vacío centrado en la ubicación con overlay ARBA
        self.chk_arba.setEnabled(True)
        self.chk_arba.setChecked(True)   # activar overlay automáticamente

        # Si no hay imagen satélite cargada, mostrar solo el overlay ARBA
        # sobre un fondo gris que representa el área
        self._show_arba_only_view(lat, lon, wms_array, bbox)

        self.statusBar().showMessage(
            f"Localización: ({lat:.5f}, {lon:.5f}) | "
            f"Bbox: {bbox[0]:.4f},{bbox[1]:.4f} → {bbox[2]:.4f},{bbox[3]:.4f} | "
            "Hacé clic en el mapa para seleccionar una parcela"
        )

    def _on_location_error(self, msg: str) -> None:
        self._show_error("No se pudo geocodificar la dirección", msg)

    def _on_toggle_arba_overlay(self, state: int) -> None:
        checked = state == Qt.CheckState.Checked.value
        if checked and hasattr(self, "_wms_array_cache"):
            self.map_widget.set_parcelas_overlay(self._wms_array_cache, alpha=0.65)
        else:
            self.map_widget.set_parcelas_overlay(None)

    def _on_map_geo_click(self, lat: float, lon: float) -> None:
        """Responde al clic georreferenciado en el mapa."""
        if self._current_bbox_wgs84 is None:
            return

        self.statusBar().showMessage(f"Consultando parcela en ({lat:.5f}, {lon:.5f})…")

        # Calcular coordenadas de píxel relativas al bbox visible
        bbox = self._current_bbox_wgs84
        lon_min, lat_min, lon_max, lat_max = bbox
        MAP_W, MAP_H = 512, 512
        px = int((lon - lon_min) / (lon_max - lon_min) * MAP_W)
        py = int((lat_max - lat) / (lat_max - lat_min) * MAP_H)

        self._pinfo_thread = QThread()
        self._pinfo_worker = ParcelInfoWorker(bbox, px, py, MAP_W, MAP_H)
        self._pinfo_worker.moveToThread(self._pinfo_thread)
        self._pinfo_thread.started.connect(self._pinfo_worker.run)
        self._pinfo_worker.finished.connect(self._on_parcel_info_received)
        self._pinfo_worker.error.connect(
            lambda e: self.statusBar().showMessage(f"No se pudo obtener info de parcela: {e}")
        )
        self._pinfo_worker.finished.connect(self._pinfo_thread.quit)
        self._pinfo_worker.error.connect(self._pinfo_thread.quit)
        self._pinfo_thread.start()

    def _on_parcel_info_received(self, info: dict) -> None:
        self._selected_parcel = info
        lat = info.get("lat", "?")
        lon = info.get("lon", "?")

        # Construir texto de info
        parts = []
        for key in ("nomenclatura", "partido", "seccion", "manzana", "parcela"):
            if key in info:
                parts.append(f"{key.capitalize()}: {info[key]}")
        if not parts:
            parts = [f"Lat: {lat}, Lon: {lon}", "(sin datos catastrales)"]

        self.lbl_parcel_info.setText("\n".join(parts))
        self.lbl_parcel_info.setStyleSheet("color: #155724; font-weight: bold;")
        self.btn_use_parcel.setEnabled(True)

        # Marcar el punto en el mapa
        label = info.get("nomenclatura", f"{lat},{lon}")
        self.map_widget.mark_parcel_click(lat, lon, label=label)

        self.statusBar().showMessage(
            f"Parcela seleccionada en ({lat}, {lon}) | "
            "Hacé clic en 'Usar esta parcela' para analizar"
        )

    def _on_use_parcel(self) -> None:
        """Abre el diálogo de descarga Sentinel-2 con el bbox de la parcela seleccionada."""
        if not self._selected_parcel or self._current_bbox_wgs84 is None:
            return
        self._on_download_image()

    # ------------------------------------------------------------------
    # Slots — Imagen y análisis
    # ------------------------------------------------------------------

    def _on_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir imagen satelital", "",
            "Archivos raster (*.tif *.tiff *.img);;Todos los archivos (*)"
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str, scl_path: str = "") -> None:
        """Carga una imagen GeoTIFF en la interfaz y muestra la preview RGB.

        Args:
            path: Ruta al GeoTIFF principal (multibanda).
            scl_path: Ruta al archivo SCL opcional. Si se provee, se carga
                      automáticamente como máscara de nubes.
        """
        try:
            from roofscan.core.ingesta.loader import load_geotiff
            data = load_geotiff(path)
        except Exception as exc:
            self._show_error("No se pudo cargar la imagen", str(exc))
            return

        self._current_file = path
        name = Path(path).name
        self.lbl_file.setText(name)
        self.lbl_file.setStyleSheet("color: #212529;")
        self.btn_run.setEnabled(True)

        # Cargar SCL si se proveyó (ej: descarga automática)
        if scl_path and Path(scl_path).exists():
            self._scl_file = scl_path
            self.lbl_scl.setText(Path(scl_path).name)
            self.lbl_scl.setStyleSheet("color: #155724;")
            self.btn_clear_scl.setEnabled(True)

        # Calcular bounds WGS84 para habilitar el clic georreferenciado
        geo_extent = self._compute_geo_extent_wgs84(data)
        if geo_extent:
            self._current_bbox_wgs84 = (
                geo_extent[0], geo_extent[2], geo_extent[1], geo_extent[3]
            )  # lon_min, lat_min, lon_max, lat_max

        self.statusBar().showMessage(
            f"Imagen: {name} | {data['count']} bandas | "
            f"{data['resolution_m'] or '?'} m/px | {data['crs']}"
        )

        # Preview RGB
        try:
            from roofscan.core.preproceso.normalizer import normalize_s2
            norm = normalize_s2(data["array"].astype(np.float32))
            if norm.shape[0] >= 3:
                self.map_widget.show_image(
                    norm, rgb_bands=(3, 2, 1), title=name,
                    geo_extent_wgs84=geo_extent,
                )
                # Restaurar overlay ARBA si estaba activo
                if self.chk_arba.isChecked() and hasattr(self, "_wms_array_cache"):
                    self.map_widget.set_parcelas_overlay(self._wms_array_cache)
        except Exception:
            pass

    def _on_download_image(self) -> None:
        """Abre el diálogo de búsqueda y descarga de Sentinel-2 desde CDSE."""
        from roofscan.gui.download_dialog import DownloadDialog
        dlg = DownloadDialog(bbox=self._current_bbox_wgs84, parent=self)
        dlg.image_ready.connect(self._on_image_downloaded)
        dlg.exec()

    def _on_image_downloaded(self, geotiff_path: str, scl_path: str) -> None:
        """Auto-carga la imagen descargada una vez que el diálogo termina."""
        self._load_file(geotiff_path, scl_path)
        self.statusBar().showMessage(
            f"Imagen descargada y cargada: {Path(geotiff_path).name}"
        )

    def _on_run_detection(self) -> None:
        if not self._current_file:
            return

        engine = "unet" if self.combo_engine.currentIndex() == 1 else "clasico"
        params = {
            "engine": engine,
            "ndvi_max": self.spin_ndvi.value(),
            "ndbi_min": self.spin_ndbi.value(),
            "ndwi_max": self.spin_ndwi.value(),
            "min_area_px": self.spin_min_area.value(),
        }
        self._set_busy(True, "Analizando imagen…")

        self._worker_thread = QThread()
        self._worker = AnalysisWorker(self._current_file, params, scl_filepath=self._scl_file)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_analysis_finished)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.progress.connect(lambda m: self.statusBar().showMessage(m))
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.error.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(lambda: self._set_busy(False, ""))
        self._worker_thread.start()

    def _on_analysis_finished(self, results: dict) -> None:
        self._results = results
        # Auto-switch to detection overlay so the user sees the result immediately
        idx_overlay = next(
            (i for i in range(self.combo_view.count())
             if self.combo_view.itemText(i) == "Overlay detección"),
            1,
        )
        self.combo_view.setCurrentIndex(idx_overlay)
        self._refresh_map_view()
        self.results_panel.show_results(results["areas"], results.get("resolution_m"))
        for btn in (self.btn_export_geotiff, self.btn_export_png,
                    self.btn_export_geojson, self.btn_export_shp, self.btn_export_csv):
            btn.setEnabled(True)
        n = results["morph"]["n_roofs"]
        total_m2 = sum(r["area_m2"] for r in results["areas"])
        # Deshabilitar vistas de índice si el motor fue U-Net
        if results.get("engine") == "unet":
            model = self.combo_view.model()
            for i, label in enumerate(["NDVI", "NDBI", "NDWI"]):
                for j in range(self.combo_view.count()):
                    if self.combo_view.itemText(j) == label:
                        model.item(j).setEnabled(False)
            if self.combo_view.currentText() in ("NDVI", "NDBI", "NDWI"):
                self.combo_view.setCurrentIndex(1)  # Overlay detección
        else:
            for j in range(self.combo_view.count()):
                self.combo_view.model().item(j).setEnabled(True)
        self.statusBar().showMessage(
            f"Detección completa | {n} techo(s) | Área total: {total_m2:,.1f} m²"
        )
        # Actualizar panel de validación con los resultados del análisis
        label = Path(self._current_file).stem if self._current_file else ""
        self.validation_panel.set_analysis_result(
            pred_area_m2=total_m2,
            image_array=results["data"]["array"],
            mask=results["morph"]["mask_clean"],
            label=label,
        )

    def _on_analysis_error(self, msg: str) -> None:
        self._show_error("Error durante la detección", msg)
        self.statusBar().showMessage("Error en el análisis.")

    def _on_view_changed(self, _: int) -> None:
        if self._results:
            self._refresh_map_view()

    # ------------------------------------------------------------------
    # Slots — Dibujo de bbox

    def _on_toggle_draw_bbox(self, checked: bool) -> None:
        """Activa o desactiva el modo de dibujo de bounding box en el mapa."""
        self.map_widget.enable_bbox_draw(checked)
        if checked:
            self.btn_draw_bbox.setText("⬚  Dibujando… (arrastrá el ratón)")
            self.statusBar().showMessage(
                "Modo dibujo activo — arrastrá el ratón sobre el mapa para definir el área."
            )
        else:
            self.btn_draw_bbox.setText("⬚  Dibujar área de análisis")

    def _on_bbox_drawn(
        self, lon_min: float, lat_min: float, lon_max: float, lat_max: float
    ) -> None:
        """Recibe el bbox dibujado, lo almacena y desactiva el modo dibujo."""
        self._current_bbox_wgs84 = (lon_min, lat_min, lon_max, lat_max)
        # Desactivar modo dibujo automáticamente
        self.btn_draw_bbox.setChecked(False)
        self.statusBar().showMessage(
            f"Área de análisis definida: "
            f"{lon_min:.4f},{lat_min:.4f} → {lon_max:.4f},{lat_max:.4f}"
        )

    # Slots — Exportación
    # ------------------------------------------------------------------

    def _on_export_geotiff(self) -> None:
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Guardar GeoTIFF", "raster_procesado.tif", "GeoTIFF (*.tif)")
        if not path:
            return
        try:
            from roofscan.core.exportacion.raster_exporter import export_geotiff
            export_geotiff(self._results["data"], Path(path).parent, filename=Path(path).stem)
            self.statusBar().showMessage(f"GeoTIFF guardado: {path}")
        except Exception as exc:
            self._show_error("No se pudo guardar el GeoTIFF", str(exc))

    def _on_export_png(self) -> None:
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Guardar PNG", "preview.png", "PNG (*.png)")
        if not path:
            return
        try:
            from roofscan.core.exportacion.raster_exporter import export_preview_png
            export_preview_png(self._results["data"], Path(path).parent, filename=Path(path).stem)
            self.statusBar().showMessage(f"PNG guardado: {path}")
        except Exception as exc:
            self._show_error("No se pudo guardar el PNG", str(exc))

    def _on_export_geojson(self) -> None:
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Guardar GeoJSON", "techos.geojson", "GeoJSON (*.geojson)")
        if not path:
            return
        try:
            from roofscan.core.calculo.geometry_merger import labels_to_geodataframe
            from roofscan.core.exportacion.geojson_exporter import export_geojson
            morph = self._results["morph"]
            data = self._results["data"]
            gdf = labels_to_geodataframe(morph["labels"], data["transform"], data["crs"], self._results["areas"])
            export_geojson(gdf, path)
            self.statusBar().showMessage(f"GeoJSON guardado: {path}")
        except Exception as exc:
            self._show_error("No se pudo guardar el GeoJSON", str(exc))

    def _on_export_shp(self) -> None:
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Shapefile", "techos.shp", "Shapefile (*.shp)"
        )
        if not path:
            return
        try:
            from roofscan.core.calculo.geometry_merger import labels_to_geodataframe
            from roofscan.core.exportacion.shp_exporter import export_shapefile
            morph = self._results["morph"]
            data = self._results["data"]
            gdf = labels_to_geodataframe(
                morph["labels"], data["transform"], data["crs"], self._results["areas"]
            )
            export_shapefile(gdf, path)
            self.statusBar().showMessage(f"Shapefile guardado: {path}")
        except Exception as exc:
            self._show_error("No se pudo guardar el Shapefile", str(exc))

    def _on_export_csv(self) -> None:
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar CSV", "techos.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            from roofscan.core.calculo.geometry_merger import labels_to_geodataframe
            from roofscan.core.exportacion.csv_exporter import export_csv
            morph = self._results["morph"]
            data = self._results["data"]
            areas = self._results["areas"]
            # Intentar computar centroides georreferenciados
            try:
                gdf = labels_to_geodataframe(
                    morph["labels"], data["transform"], data["crs"], areas
                )
            except Exception:
                gdf = None
            p = Path(path)
            export_csv(areas, output_dir=p.parent, filename=p.name, gdf=gdf)
            self.statusBar().showMessage(f"CSV guardado: {path}")
        except Exception as exc:
            self._show_error("No se pudo guardar el CSV", str(exc))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _start_retrain(self) -> None:
        """Inicia el reentrenamiento del modelo U-Net en background."""
        self.validation_panel.update_retrain_state(running=True)
        self.statusBar().showMessage("Reentrenamiento iniciado…")

        self._retrain_thread = QThread()
        self._retrain_worker = RetrainWorker(epochs=20, batch_size=4)
        self._retrain_worker.moveToThread(self._retrain_thread)
        self._retrain_thread.started.connect(self._retrain_worker.run)
        self._retrain_worker.finished.connect(self._on_retrain_finished)
        self._retrain_worker.error.connect(self._on_retrain_error)
        self._retrain_worker.progress.connect(lambda m: self.statusBar().showMessage(m))
        self._retrain_worker.finished.connect(self._retrain_thread.quit)
        self._retrain_worker.error.connect(self._retrain_thread.quit)
        self._retrain_thread.start()

    def _on_retrain_finished(self, history: dict) -> None:
        self.validation_panel.show_retrain_result(history)
        best = history.get("best_val_loss", float("nan"))
        self.statusBar().showMessage(
            f"Reentrenamiento completo | mejor val_loss={best:.4f}"
        )

    def _on_retrain_error(self, msg: str) -> None:
        self.validation_panel.show_retrain_error(msg)
        self._show_error("Error en reentrenamiento", msg)
        self.statusBar().showMessage("Error en el reentrenamiento.")

    def _show_arba_only_view(self, lat, lon, wms_array, bbox) -> None:
        """Muestra el overlay ARBA sobre un fondo oscuro cuando no hay imagen satélite."""
        lon_min, lat_min, lon_max, lat_max = bbox
        H, W = wms_array.shape[:2]

        # Fondo gris oscuro como placeholder de imagen satélite
        fake_base = np.full((3, H, W), 0.12, dtype=np.float32)
        geo_extent = (lon_min, lon_max, lat_min, lat_max)

        self.map_widget.show_image(
            fake_base,
            rgb_bands=(1, 2, 3),
            title=f"Parcelas ARBA — ({lat:.4f}, {lon:.4f})",
            geo_extent_wgs84=geo_extent,
        )
        self.map_widget.set_parcelas_overlay(wms_array, alpha=0.85)

    def _compute_geo_extent_wgs84(self, data: dict) -> tuple | None:
        """Calcula el extent WGS84 (lon_min, lon_max, lat_min, lat_max) de un raster."""
        try:
            from pyproj import Transformer
            from rasterio.transform import array_bounds

            _, h, w = data["array"].shape
            bounds = data.get("bounds")
            if bounds is None:
                from rasterio.transform import array_bounds as ab
                b = ab(h, w, data["transform"])
                left, bottom, right, top = b.left, b.bottom, b.right, b.top
            elif isinstance(bounds, tuple):
                left, bottom, right, top = bounds
            else:
                left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top

            crs = data["crs"]
            if "4326" in str(crs):
                return (left, right, bottom, top)

            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon_min, lat_min = transformer.transform(left, bottom)
            lon_max, lat_max = transformer.transform(right, top)
            return (lon_min, lon_max, lat_min, lat_max)
        except Exception as exc:
            log.warning("No se pudo calcular extent WGS84: %s", exc)
            return None

    def _refresh_map_view(self) -> None:
        if not self._results:
            return
        arr = self._results["data"]["array"]
        view = self.combo_view.currentText()
        geo_ext = self._compute_geo_extent_wgs84(self._results["data"])
        try:
            if view == "RGB True Color":
                self.map_widget.show_image(arr, rgb_bands=(3, 2, 1), geo_extent_wgs84=geo_ext)
            elif view == "Overlay detección":
                self.map_widget.show_detection_overlay(
                    arr, self._results["morph"]["mask_clean"], geo_extent_wgs84=geo_ext
                )
            elif view == "NDVI":
                idx_arr = self._results["detection"].get("ndvi")
                if idx_arr is not None:
                    self.map_widget.show_index(idx_arr, "NDVI")
            elif view == "NDBI":
                idx_arr = self._results["detection"].get("ndbi")
                if idx_arr is not None:
                    self.map_widget.show_index(idx_arr, "NDBI")
            elif view == "NDWI":
                idx_arr = self._results["detection"].get("ndwi")
                if idx_arr is not None:
                    self.map_widget.show_index(idx_arr, "NDWI")
            # Restaurar overlay si está activo
            if self.chk_arba.isChecked() and hasattr(self, "_wms_array_cache"):
                self.map_widget.set_parcelas_overlay(self._wms_array_cache)
        except Exception as exc:
            log.warning("No se pudo actualizar la vista: %s", exc)

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self.btn_run.setEnabled(not busy)
        self.btn_open.setEnabled(not busy)
        self.btn_search.setEnabled(not busy)
        if busy:
            self.progress_bar.show()
            self.statusBar().showMessage(message)
        else:
            self.progress_bar.hide()

    def _show_error(self, title: str, detail: str) -> None:
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(detail)
        box.exec()

    @staticmethod
    def _labeled_spin(layout, label, mn, mx, default, step, tip="") -> QDoubleSpinBox:
        lbl = QLabel(label)
        lbl.setFont(QFont("Segoe UI", 8))
        spin = QDoubleSpinBox()
        spin.setRange(mn, mx)
        spin.setSingleStep(step)
        spin.setDecimals(2)
        spin.setValue(default)
        spin.setFont(QFont("Segoe UI", 9))
        if tip:
            spin.setToolTip(tip)
        layout.addWidget(lbl)
        layout.addWidget(spin)
        return spin

    @staticmethod
    def _labeled_int_spin(layout, label, mn, mx, default, tip="") -> QSpinBox:
        lbl = QLabel(label)
        lbl.setFont(QFont("Segoe UI", 8))
        spin = QSpinBox()
        spin.setRange(mn, mx)
        spin.setValue(default)
        spin.setFont(QFont("Segoe UI", 9))
        if tip:
            spin.setToolTip(tip)
        layout.addWidget(lbl)
        layout.addWidget(spin)
        return spin
