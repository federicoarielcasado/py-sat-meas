"""Widget de visualización de imágenes raster embebido en PyQt6.

Novedades respecto a Sprint 3:
- Almacena el extent geográfico (WGS84) de la imagen mostrada.
- Soporta overlay semitransparente de la capa de parcelas CartoARBA (WMS).
- Emite la señal ``geo_clicked(lat, lon)`` cuando el usuario hace clic
  sobre el canvas, expresada en coordenadas geográficas.
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MapWidget(QWidget):
    """Canvas matplotlib embebido en Qt para visualización de rasters.

    Signals:
        geo_clicked (float, float): Emitida al hacer clic en el canvas.
            Los argumentos son ``(lat, lon)`` en WGS84.

    Args:
        parent: Widget padre de Qt.
    """

    geo_clicked = pyqtSignal(float, float)   # lat, lon

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = None
        self._mask = None
        self._geo_extent_wgs84 = None   # (lon_min, lon_max, lat_min, lat_max)
        self._wms_overlay = None        # array RGBA del overlay de parcelas

        self.figure = Figure(figsize=(6, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#1a1a2e")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        # Conectar el evento de clic de matplotlib
        self._cid_click = self.canvas.mpl_connect(
            "button_press_event", self._on_mpl_click
        )

        self._show_placeholder()

    # ------------------------------------------------------------------
    # API pública — visualización de imagen satélite
    # ------------------------------------------------------------------

    def show_image(
        self,
        array: np.ndarray,
        rgb_bands: tuple[int, int, int] = (3, 2, 1),
        title: str = "Imagen satelital",
        geo_extent_wgs84: tuple | None = None,
    ) -> None:
        """Muestra el array raster como imagen RGB.

        Args:
            array: Array float32 ``(bandas, H, W)`` normalizado [0, 1].
            rgb_bands: Índices 1-based de las bandas R, G, B.
            title: Título del panel.
            geo_extent_wgs84: Extensión geográfica ``(lon_min, lon_max, lat_min, lat_max)``
                en WGS84. Necesario para el clic georreferenciado y el overlay ARBA.
        """
        self._data = array
        self._geo_extent_wgs84 = geo_extent_wgs84
        rgb = self._to_rgb(array, rgb_bands)
        self._draw_base(rgb, title)
        if self._wms_overlay is not None:
            self._draw_wms_overlay()
        self.canvas.draw()

    def show_detection_overlay(
        self,
        array: np.ndarray,
        mask: np.ndarray,
        rgb_bands: tuple[int, int, int] = (3, 2, 1),
        geo_extent_wgs84: tuple | None = None,
    ) -> None:
        """Superpone la máscara de detección sobre la imagen RGB.

        Args:
            array: Array float32 normalizado.
            mask: Array booleano 2D ``(H, W)``. ``True`` = techo detectado.
            rgb_bands: Índices de bandas para el fondo RGB.
            geo_extent_wgs84: Extensión geográfica opcional.
        """
        self._data = array
        self._mask = mask
        if geo_extent_wgs84 is not None:
            self._geo_extent_wgs84 = geo_extent_wgs84

        rgb = self._to_rgb(array, rgb_bands)
        self._draw_base(rgb, f"Detección — {mask.sum():,} px detectados")

        # Overlay detección: techos en naranja semitransparente
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask, 0] = 1.0
        overlay[mask, 1] = 0.65
        overlay[mask, 2] = 0.0
        overlay[mask, 3] = 0.55
        self.ax.imshow(overlay, interpolation="nearest", aspect="auto")

        if self._wms_overlay is not None:
            self._draw_wms_overlay()
        self.canvas.draw()

    def show_index(self, index_array: np.ndarray, label: str = "Índice") -> None:
        """Muestra un índice espectral (NDVI, NDBI, etc.) en escala de color."""
        self.ax.clear()
        im = self.ax.imshow(
            index_array, cmap="RdYlGn", vmin=-1, vmax=1, interpolation="nearest"
        )
        self.figure.colorbar(im, ax=self.ax, fraction=0.03, pad=0.02)
        self.ax.set_title(label, fontsize=9, color="white", pad=4)
        self.ax.axis("off")
        self.figure.patch.set_facecolor("#1a1a2e")
        self.canvas.draw()

    # ------------------------------------------------------------------
    # API pública — overlay de parcelas CartoARBA
    # ------------------------------------------------------------------

    def set_parcelas_overlay(
        self,
        wms_array: np.ndarray | None,
        alpha: float = 0.65,
    ) -> None:
        """Carga o quita el overlay de parcelas ARBA.

        Args:
            wms_array: Array RGBA ``(H, W, 4)`` uint8 con la imagen WMS.
                       Pasar ``None`` para quitar el overlay.
            alpha: Opacidad del overlay (0.0 transparente – 1.0 opaco).
        """
        if wms_array is None:
            self._wms_overlay = None
        else:
            # Convertir a float [0,1] y aplicar alpha deseado
            arr = wms_array.astype(np.float32) / 255.0
            # Dónde la imagen es casi blanca (fondo WMS) → transparentar
            is_white = (arr[..., :3] > 0.95).all(axis=-1)
            arr[is_white, 3] = 0.0
            arr[~is_white, 3] = alpha
            self._wms_overlay = arr

        # Redibujar si ya hay imagen base cargada
        if self._data is not None:
            self._redraw_current()

    def mark_parcel_click(self, lat: float, lon: float, label: str = "") -> None:
        """Dibuja un marcador en el punto geográfico dado.

        Args:
            lat: Latitud del punto en WGS84.
            lon: Longitud del punto en WGS84.
            label: Texto opcional a mostrar junto al marcador.
        """
        if self._geo_extent_wgs84 is None or self._data is None:
            return

        lon_min, lon_max, lat_min, lat_max = self._geo_extent_wgs84
        _, img_h, img_w = self._data.shape

        # Convertir geo → píxel
        px = (lon - lon_min) / (lon_max - lon_min) * img_w
        py = (lat_max - lat) / (lat_max - lat_min) * img_h

        # Quitar marcadores anteriores y añadir nuevo
        for artist in list(self.ax.lines) + list(self.ax.texts):
            if getattr(artist, "_roofscan_marker", False):
                artist.remove()

        marker, = self.ax.plot(px, py, "r+", markersize=14, markeredgewidth=2)
        marker._roofscan_marker = True

        if label:
            txt = self.ax.text(
                px + 5, py - 5, label,
                color="red", fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            )
            txt._roofscan_marker = True

        self.canvas.draw()

    def clear(self) -> None:
        """Limpia el canvas y muestra el placeholder."""
        self._data = None
        self._mask = None
        self._geo_extent_wgs84 = None
        self._wms_overlay = None
        self._show_placeholder()

    # ------------------------------------------------------------------
    # Slots privados
    # ------------------------------------------------------------------

    def _on_mpl_click(self, event) -> None:
        """Convierte el clic matplotlib (píxel) a coordenadas geográficas y emite señal."""
        if event.inaxes is None or self._geo_extent_wgs84 is None or self._data is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        lon_min, lon_max, lat_min, lat_max = self._geo_extent_wgs84
        _, img_h, img_w = self._data.shape

        # event.xdata, event.ydata son coordenadas de imagen (píxeles)
        px = event.xdata
        py = event.ydata

        lon = lon_min + (px / img_w) * (lon_max - lon_min)
        lat = lat_max - (py / img_h) * (lat_max - lat_min)

        # Verificar que el punto esté dentro del extent
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            self.geo_clicked.emit(round(lat, 6), round(lon, 6))

    # ------------------------------------------------------------------
    # Helpers de dibujo internos
    # ------------------------------------------------------------------

    def _draw_base(self, rgb: np.ndarray, title: str) -> None:
        self.ax.clear()
        self.ax.imshow(rgb, interpolation="nearest", aspect="auto")
        self.ax.set_title(title, fontsize=9, color="white", pad=4)
        self.ax.axis("off")
        self.figure.patch.set_facecolor("#1a1a2e")
        self.ax.set_facecolor("#1a1a2e")

    def _draw_wms_overlay(self) -> None:
        """Dibuja el overlay WMS sobre la imagen base ya renderizada."""
        if self._wms_overlay is None:
            return
        self.ax.imshow(self._wms_overlay, interpolation="nearest", aspect="auto")

    def _redraw_current(self) -> None:
        """Redibuja la vista actual (detecta cuál está activa)."""
        if self._data is None:
            return
        if self._mask is not None:
            self.show_detection_overlay(self._data, self._mask)
        else:
            self.show_image(self._data)

    def _show_placeholder(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor("#1a1a2e")
        self.ax.text(
            0.5, 0.5,
            "Cargá una imagen o buscá una dirección para comenzar",
            ha="center", va="center",
            color="#6c757d", fontsize=10,
            transform=self.ax.transAxes,
        )
        self.ax.axis("off")
        self.figure.patch.set_facecolor("#1a1a2e")
        self.canvas.draw()

    @staticmethod
    def _to_rgb(array: np.ndarray, rgb_bands: tuple[int, int, int]) -> np.ndarray:
        r, g, b = rgb_bands
        rgb = np.stack([array[r - 1], array[g - 1], array[b - 1]], axis=-1)
        rgb = np.nan_to_num(rgb, nan=0.0)
        lo, hi = np.nanpercentile(rgb, [2, 98])
        if hi > lo:
            rgb = (rgb - lo) / (hi - lo)
        return np.clip(rgb, 0.0, 1.0)
