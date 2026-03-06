"""Widget de visualización de imágenes raster embebido en PyQt6.

Funcionalidades:
- Visualización de imagen satelital RGB, overlay de detección e índices espectrales.
- Overlay WMS rasterizado de parcelas CartoARBA (coexiste con la capa vectorial).
- Capa vectorial de parcelas: polígonos georeferenciados dibujados sobre la imagen con
  posición exacta usando la transformada affine. Soporta 140k+ parcelas con filtrado
  automático al viewport actual.
- Zoom con scroll del mouse (centrado en el cursor). Reset con ``reset_zoom()``.
- Clic georreferenciado: emite ``geo_clicked(lat, lon)`` y además busca la parcela
  vectorial bajo el cursor (emite ``parcel_vector_clicked(idx)``).
- Modo dibujo de bbox: al soltar emite ``bbox_selected`` (WGS84) y, si hay parcelas
  cargadas, también ``parcels_vector_selected(indices)`` con los índices de las
  parcelas que intersectan el rectángulo dibujado.
"""

import logging

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

log = logging.getLogger(__name__)

# Máximo de líneas de parcelas a dibujar por renderizado (performance).
_MAX_PARCELA_LINES = 12_000


class MapWidget(QWidget):
    """Canvas matplotlib embebido en Qt para visualización de rasters.

    Signals:
        geo_clicked (float, float): Emitida al hacer clic en el canvas
            fuera del modo dibujo. Argumentos: ``(lat, lon)`` en WGS84.
        bbox_selected (float, float, float, float): Emitida al completar
            un rectángulo en modo dibujo.
            Argumentos: ``(lon_min, lat_min, lon_max, lat_max)`` en WGS84.
        parcel_vector_clicked (int): Índice de la parcela vectorial bajo el
            cursor al hacer clic (``-1`` si no hay ninguna).
        parcels_vector_selected (list): Lista de índices de parcelas que
            intersectan el bbox dibujado. Lista vacía si no hay parcelas cargadas.

    Args:
        parent: Widget padre de Qt.
    """

    geo_clicked               = pyqtSignal(float, float)
    bbox_selected             = pyqtSignal(float, float, float, float)
    parcel_vector_clicked     = pyqtSignal(int)
    parcels_vector_selected   = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Estado de imagen base ---
        self._data: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._geo_extent_wgs84: tuple | None = None   # (lon_min, lon_max, lat_min, lat_max)
        self._img_transform = None                    # rasterio affine transform
        self._img_crs: str | None = None              # CRS de la imagen (e.g. "EPSG:32721")

        # --- Overlay WMS (raster de CartoARBA) ---
        self._wms_overlay: np.ndarray | None = None   # array RGBA float [0,1]

        # --- Capa vectorial de parcelas ---
        self._gdf_parcelas_repr = None    # GDF reprojected to img_crs
        self._parcela_lines_px: list | None = None   # coords pixel por parcela
        self._selected_ids: set = set()              # índices seleccionados

        # --- Modo dibujo de bbox ---
        self._draw_mode: bool = False
        self._selector = None

        # --- Matplotlib ---
        self.figure = Figure(figsize=(6, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#1a1a2e")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        # Conectar eventos matplotlib
        self._cid_click  = self.canvas.mpl_connect("button_press_event", self._on_mpl_click)
        self._cid_scroll = self.canvas.mpl_connect("scroll_event", self._on_scroll)

        self._show_placeholder()

    # ------------------------------------------------------------------
    # API pública — georeferenciación de la imagen
    # ------------------------------------------------------------------

    def set_image_georef(self, transform, crs: str) -> None:
        """Almacena la transformada affine y el CRS de la imagen cargada.

        Debe llamarse cada vez que se carga una imagen nueva, antes (o justo
        después) de ``show_image()``. Necesario para la capa vectorial y el
        hit-testing preciso de parcelas.

        Args:
            transform: Transformada affine rasterio (``data["transform"]``).
            crs: CRS de la imagen como string (e.g. ``"EPSG:32721"``).
        """
        self._img_transform = transform
        self._img_crs = str(crs)

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
        if geo_extent_wgs84 is not None:
            self._geo_extent_wgs84 = geo_extent_wgs84
        rgb = self._to_rgb(array, rgb_bands)
        self._draw_base(rgb, title)
        if self._wms_overlay is not None:
            self._draw_wms_overlay()
        self._draw_parcelas_vector()
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
        self._draw_parcelas_vector()
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
    # API pública — overlay WMS de parcelas CartoARBA
    # ------------------------------------------------------------------

    def set_parcelas_overlay(
        self,
        wms_array: np.ndarray | None,
        alpha: float = 0.65,
    ) -> None:
        """Carga o quita el overlay WMS rasterizado de parcelas ARBA.

        Coexiste con la capa vectorial: el WMS se dibuja primero y los
        polígonos vectoriales se superponen encima.

        Args:
            wms_array: Array RGBA ``(H, W, 4)`` uint8 con la imagen WMS.
                       Pasar ``None`` para quitar el overlay.
            alpha: Opacidad del overlay (0.0 transparente – 1.0 opaco).
        """
        if wms_array is None:
            self._wms_overlay = None
        else:
            arr = wms_array.astype(np.float32) / 255.0
            is_white = (arr[..., :3] > 0.95).all(axis=-1)
            arr[is_white, 3] = 0.0
            arr[~is_white, 3] = alpha
            self._wms_overlay = arr

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

        px = (lon - lon_min) / (lon_max - lon_min) * img_w
        py = (lat_max - lat) / (lat_max - lat_min) * img_h

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
        self.enable_bbox_draw(False)
        self._data = None
        self._mask = None
        self._geo_extent_wgs84 = None
        self._wms_overlay = None
        self._img_transform = None
        self._img_crs = None
        self._gdf_parcelas_repr = None
        self._parcela_lines_px = None
        self._selected_ids = set()
        self._show_placeholder()

    # ------------------------------------------------------------------
    # API pública — capa vectorial de parcelas
    # ------------------------------------------------------------------

    def set_parcelas_vector(self, gdf) -> None:
        """Carga una capa vectorial de parcelas sobre la imagen.

        Reproyecta el GeoDataFrame al CRS de la imagen y precomputa las
        coordenadas píxel de cada polígono para un renderizado eficiente.
        Solo dibuja las parcelas dentro del extent visible (hasta
        ``_MAX_PARCELA_LINES`` líneas).

        Requiere haber llamado a ``set_image_georef()`` primero.

        Args:
            gdf: GeoDataFrame con polígonos de parcelas (cualquier CRS).
        """
        if self._img_transform is None or self._img_crs is None:
            log.warning("set_parcelas_vector: imagen no georeferenciada — cargá una imagen primero.")
            return

        try:
            from pyproj import CRS as ProjCRS
        except ImportError:
            log.warning("pyproj no disponible para reproyectar parcelas vectoriales.")
            return

        try:
            img_crs_obj = ProjCRS.from_user_input(self._img_crs)
            gdf_repr = gdf.to_crs(img_crs_obj).reset_index(drop=True)
        except Exception as exc:
            log.warning("No se pudo reproyectar las parcelas al CRS de la imagen: %s", exc)
            return

        self._gdf_parcelas_repr = gdf_repr
        self._selected_ids = set()

        # Precomputar coordenadas pixel de cada polígono exterior
        inv_T = ~self._img_transform
        lines_px = []
        for geom in gdf_repr.geometry:
            if geom is None or geom.is_empty:
                lines_px.append([])
                continue
            polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
            parts = []
            for poly in polys:
                coords = np.array(poly.exterior.coords)
                cols = inv_T.a * coords[:, 0] + inv_T.b * coords[:, 1] + inv_T.c
                rows = inv_T.d * coords[:, 0] + inv_T.e * coords[:, 1] + inv_T.f
                parts.append(np.column_stack([cols, rows]))
            lines_px.append(parts)

        self._parcela_lines_px = lines_px
        log.info("Parcelas vectoriales cargadas: %d polígonos", len(gdf_repr))

        if self._data is not None:
            self._redraw_current()

    def clear_parcelas_vector(self) -> None:
        """Quita la capa vectorial de parcelas del mapa."""
        self._gdf_parcelas_repr = None
        self._parcela_lines_px = None
        self._selected_ids = set()
        if self._data is not None:
            self._redraw_current()

    def select_parcelas(self, indices: list[int]) -> None:
        """Resalta las parcelas dadas en el mapa.

        Args:
            indices: Lista de índices (posición en el GDF) a resaltar.
        """
        self._selected_ids = set(indices)
        if self._data is not None:
            self._redraw_current()

    # ------------------------------------------------------------------
    # API pública — dibujo interactivo de bbox
    # ------------------------------------------------------------------

    def enable_bbox_draw(self, active: bool) -> None:
        """Activa o desactiva el modo de dibujo de bounding box.

        Args:
            active: ``True`` para activar el selector, ``False`` para desactivarlo.
        """
        self._draw_mode = active

        if active and self._data is not None:
            from matplotlib.widgets import RectangleSelector
            self._selector = RectangleSelector(
                self.ax,
                self._on_rect_select,
                useblit=True,
                button=[1],
                minspanx=5,
                minspany=5,
                spancoords="pixels",
                props=dict(edgecolor="cyan", facecolor="cyan", alpha=0.15, linewidth=1.5),
                interactive=False,
            )
        else:
            if self._selector is not None:
                self._selector.set_active(False)
                self._selector = None
            self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # API pública — zoom
    # ------------------------------------------------------------------

    def reset_zoom(self) -> None:
        """Restaura el zoom para mostrar la imagen completa."""
        if self._data is None:
            return
        _, H, W = self._data.shape
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)   # eje Y invertido (origen arriba)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Slots privados — eventos matplotlib
    # ------------------------------------------------------------------

    def _on_mpl_click(self, event) -> None:
        """Clic en canvas: convierte a WGS84, hace hit-test de parcelas."""
        if self._draw_mode:
            return
        if event.inaxes is None or self._data is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        px, py = event.xdata, event.ydata

        # --- Hit-test parcela vectorial ---
        idx = self._hit_test_parcel(px, py)
        if idx >= 0:
            self._selected_ids = {idx}
            self._redraw_current()
            self.parcel_vector_clicked.emit(idx)

        # --- Señal geo_clicked (WGS84) ---
        if self._geo_extent_wgs84 is None:
            return
        lon_min, lon_max, lat_min, lat_max = self._geo_extent_wgs84
        _, img_h, img_w = self._data.shape
        lon = lon_min + (px / img_w) * (lon_max - lon_min)
        lat = lat_max - (py / img_h) * (lat_max - lat_min)
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            self.geo_clicked.emit(round(lat, 6), round(lon, 6))

    def _on_rect_select(self, eclick, erelease) -> None:
        """Rectángulo dibujado: emite bbox WGS84 y parcelas seleccionadas."""
        if self._data is None:
            return
        if eclick.xdata is None or erelease.xdata is None:
            return

        x1 = min(eclick.xdata, erelease.xdata)
        x2 = max(eclick.xdata, erelease.xdata)
        y1 = min(eclick.ydata, erelease.ydata)
        y2 = max(eclick.ydata, erelease.ydata)

        # --- Hit-test parcelas en el bbox (pixel space) ---
        indices = self._hit_test_parcels_bbox(x1, y1, x2, y2)
        if indices:
            self._selected_ids = set(indices)
            self._redraw_current()
        self.parcels_vector_selected.emit(indices)

        # --- Señal bbox_selected (WGS84) ---
        if self._geo_extent_wgs84 is None:
            return
        lon_min_e, lon_max_e, lat_min_e, lat_max_e = self._geo_extent_wgs84
        _, img_h, img_w = self._data.shape

        lon1 = lon_min_e + (x1 / img_w) * (lon_max_e - lon_min_e)
        lon2 = lon_min_e + (x2 / img_w) * (lon_max_e - lon_min_e)
        lat2 = lat_max_e - (y1 / img_h) * (lat_max_e - lat_min_e)
        lat1 = lat_max_e - (y2 / img_h) * (lat_max_e - lat_min_e)

        self.bbox_selected.emit(
            round(lon1, 6), round(lat1, 6),
            round(lon2, 6), round(lat2, 6),
        )

    def _on_scroll(self, event) -> None:
        """Zoom con scroll del mouse, centrado en el cursor."""
        if event.inaxes is None or self._data is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        scale = 0.82 if event.button == "up" else 1.0 / 0.82
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x, y = event.xdata, event.ydata

        self.ax.set_xlim([x + (xl - x) * scale for xl in xlim])
        self.ax.set_ylim([y + (yl - y) * scale for yl in ylim])
        self.canvas.draw_idle()

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
        """Dibuja el overlay WMS (raster) sobre la imagen base."""
        if self._wms_overlay is None:
            return
        self.ax.imshow(self._wms_overlay, interpolation="nearest", aspect="auto")

    def _draw_parcelas_vector(self) -> None:
        """Dibuja los polígonos de parcelas como LineCollection.

        Solo renderiza parcelas dentro del viewport actual. Las parcelas
        seleccionadas se dibujan en cian con borde más grueso.
        """
        if self._parcela_lines_px is None or self._gdf_parcelas_repr is None:
            return

        try:
            from matplotlib.collections import LineCollection
            from shapely.geometry import box as shapely_box
        except ImportError:
            return

        # Obtener viewport actual en píxeles
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xmin, xmax = sorted(xlim)
        ymin, ymax = sorted(ylim)

        # Convertir viewport a image CRS para filtrar con sindex
        visible_ids: list[int] | None = None
        if self._img_transform is not None:
            try:
                T = self._img_transform
                corners_px = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
                xs, ys = [], []
                for cpx, cpy in corners_px:
                    xs.append(T.a * cpx + T.b * cpy + T.c)
                    ys.append(T.d * cpx + T.e * cpy + T.f)
                view_box = shapely_box(min(xs), min(ys), max(xs), max(ys))
                visible_ids = list(
                    self._gdf_parcelas_repr.sindex.intersection(view_box.bounds)
                )
            except Exception:
                visible_ids = None

        if visible_ids is None:
            visible_ids = list(range(len(self._parcela_lines_px)))

        unsel_lines = []
        sel_lines   = []
        drawn = 0

        for i in visible_ids:
            if drawn >= _MAX_PARCELA_LINES:
                break
            parts = self._parcela_lines_px[i]
            if not parts:
                continue
            if i in self._selected_ids:
                sel_lines.extend(parts)
            else:
                unsel_lines.extend(parts)
            drawn += 1

        # Parcelas normales: amarillo suave, trazo fino
        if unsel_lines:
            lc = LineCollection(
                unsel_lines,
                colors="#FFD700", linewidths=0.6, alpha=0.75, zorder=3,
            )
            lc._roofscan_parcelas = True
            self.ax.add_collection(lc)

        # Parcelas seleccionadas: cian, trazo grueso
        if sel_lines:
            lc_sel = LineCollection(
                sel_lines,
                colors="#00FFFF", linewidths=2.5, alpha=0.95, zorder=4,
            )
            lc_sel._roofscan_parcelas = True
            self.ax.add_collection(lc_sel)

        if drawn >= _MAX_PARCELA_LINES:
            log.debug(
                "_draw_parcelas_vector: límite de %d líneas alcanzado — hacé zoom para ver más detalle.",
                _MAX_PARCELA_LINES,
            )

    def _redraw_current(self) -> None:
        """Redibuja la vista actual conservando el geo_extent."""
        if self._data is None:
            return
        if self._mask is not None:
            self.show_detection_overlay(
                self._data, self._mask, geo_extent_wgs84=self._geo_extent_wgs84
            )
        else:
            self.show_image(self._data, geo_extent_wgs84=self._geo_extent_wgs84)

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

    # ------------------------------------------------------------------
    # Hit-testing — parcelas vectoriales
    # ------------------------------------------------------------------

    def _hit_test_parcel(self, px: float, py: float) -> int:
        """Devuelve el índice de la parcela que contiene el píxel (px, py), o -1."""
        if self._gdf_parcelas_repr is None or self._img_transform is None:
            return -1
        try:
            from shapely.geometry import Point
            T = self._img_transform
            x_crs = T.a * px + T.b * py + T.c
            y_crs = T.d * px + T.e * py + T.f
            pt = Point(x_crs, y_crs)
            candidates = list(self._gdf_parcelas_repr.sindex.intersection(pt.bounds))
            for i in candidates:
                geom = self._gdf_parcelas_repr.geometry.iloc[i]
                if geom is not None and not geom.is_empty and geom.contains(pt):
                    return i
        except Exception:
            pass
        return -1

    def _hit_test_parcels_bbox(
        self, x1_px: float, y1_px: float, x2_px: float, y2_px: float
    ) -> list[int]:
        """Devuelve índices de parcelas que intersectan el bbox en píxeles."""
        if self._gdf_parcelas_repr is None or self._img_transform is None:
            return []
        try:
            from shapely.geometry import box as shapely_box
            T = self._img_transform
            corners = [(x1_px, y1_px), (x2_px, y1_px), (x2_px, y2_px), (x1_px, y2_px)]
            xs = [T.a * cpx + T.b * cpy + T.c for cpx, cpy in corners]
            ys = [T.d * cpx + T.e * cpy + T.f for cpx, cpy in corners]
            sel_box = shapely_box(min(xs), min(ys), max(xs), max(ys))
            candidates = list(
                self._gdf_parcelas_repr.sindex.intersection(sel_box.bounds)
            )
            return [
                i for i in candidates
                if self._gdf_parcelas_repr.geometry.iloc[i].intersects(sel_box)
            ]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Helpers estáticos
    # ------------------------------------------------------------------

    @staticmethod
    def _to_rgb(array: np.ndarray, rgb_bands: tuple[int, int, int]) -> np.ndarray:
        r, g, b = rgb_bands
        rgb = np.stack([array[r - 1], array[g - 1], array[b - 1]], axis=-1)
        rgb = np.nan_to_num(rgb, nan=0.0)
        lo, hi = np.nanpercentile(rgb, [2, 98])
        if hi > lo:
            rgb = (rgb - lo) / (hi - lo)
        return np.clip(rgb, 0.0, 1.0)
