"""Carga de imágenes raster locales (GeoTIFF y formatos compatibles con GDAL).

Permite que el usuario provea su propia imagen satelital ya descargada,
sin necesidad de credenciales CDSE.

Uso típico::

    from roofscan.core.ingesta.loader import load_geotiff

    data = load_geotiff("/ruta/a/imagen.tif")
    print(data["crs"])          # "EPSG:32720"
    print(data["resolution_m"]) # 10.0
    print(data["array"].shape)  # (bandas, alto, ancho)
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Resolución en metros/px para detectar el sensor por heurística
_SENSOR_RESOLUTION_MAP = {
    "Sentinel-2": 10.0,
    "Landsat-8/9": 30.0,
    "SPOT-6/7": 1.5,
    "Pleiades": 0.5,
}

# Tolerancia para la comparación de resoluciones (metros)
_RES_TOLERANCE_M = 1.0


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def load_geotiff(filepath: str | Path) -> dict[str, Any]:
    """Carga un archivo GeoTIFF (o cualquier formato raster compatible con GDAL).

    Args:
        filepath: Ruta al archivo raster. Puede ser relativa o absoluta.

    Returns:
        Diccionario con los siguientes campos:

        - ``array`` (:class:`numpy.ndarray`): Array NumPy de forma ``(bandas, alto, ancho)``.
        - ``crs`` (str): Sistema de referencia de coordenadas (ej. ``"EPSG:32720"``).
        - ``transform``: Objeto :class:`rasterio.transform.Affine` con la georreferenciación.
        - ``bounds``: :class:`rasterio.coords.BoundingBox` con las coordenadas del extent.
        - ``resolution_m`` (float | None): Resolución espacial en metros/px (None si no es métrico).
        - ``nodata`` (float | None): Valor de no-dato definido en el archivo.
        - ``count`` (int): Número de bandas.
        - ``dtype`` (str): Tipo de dato de las bandas (ej. ``"uint16"``).
        - ``sensor`` (str): Sensor estimado por heurística (ej. ``"Sentinel-2"`` o ``"desconocido"``).
        - ``filepath`` (str): Ruta absoluta al archivo cargado.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta indicada.
        ValueError: Si el archivo no es un raster válido o no tiene CRS definido.
        ImportError: Si rasterio no está instalado.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "La librería 'rasterio' no está instalada. "
            "Ejecutá: pip install rasterio"
        ) from exc

    filepath = Path(filepath).resolve()

    if not filepath.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {filepath}\n"
            "Verificá la ruta e intentá de nuevo."
        )

    log.info("Cargando raster: %s", filepath)

    try:
        with rasterio.open(filepath) as src:
            crs = src.crs
            transform = src.transform
            bounds = src.bounds
            nodata = src.nodata
            count = src.count
            dtype = src.dtypes[0]
            array = src.read()
    except Exception as exc:
        raise ValueError(
            f"No se pudo leer el archivo como raster: {filepath}\n"
            f"Detalle: {exc}\n"
            "Asegurate de que sea un GeoTIFF u otro formato compatible con GDAL."
        ) from exc

    if crs is None:
        raise ValueError(
            f"El archivo no tiene CRS (sistema de referencia de coordenadas) definido: {filepath}\n"
            "Para usar el archivo en RoofScan, primero asignale un CRS con QGIS o gdal_translate."
        )

    resolution_m = _compute_resolution_m(transform, crs)
    sensor = detect_sensor({"resolution_m": resolution_m, "count": count})

    log.info(
        "Raster cargado | CRS=%s | bandas=%d | res=%.1fm | sensor=%s",
        crs.to_string(), count, resolution_m if resolution_m else 0, sensor,
    )

    if resolution_m is not None and resolution_m > 50:
        log.warning(
            "Resolución %.1f m/px puede ser insuficiente para detectar techos pequeños.", resolution_m
        )

    return {
        "array": array,
        "crs": crs.to_string(),
        "transform": transform,
        "bounds": bounds,
        "resolution_m": resolution_m,
        "nodata": nodata,
        "count": count,
        "dtype": dtype,
        "sensor": sensor,
        "filepath": str(filepath),
    }


def detect_sensor(metadata: dict[str, Any]) -> str:
    """Estima el sensor a partir de los metadatos del raster.

    La detección es heurística: se basa en la resolución espacial y la
    cantidad de bandas. No es infalible, pero sirve para orientar al usuario.

    Args:
        metadata: Dict con al menos ``resolution_m`` (float | None) y ``count`` (int).

    Returns:
        Nombre del sensor estimado, o ``"desconocido"`` si no se puede determinar.
    """
    res = metadata.get("resolution_m")
    if res is None:
        return "desconocido"

    for sensor_name, expected_res in _SENSOR_RESOLUTION_MAP.items():
        # Comparación estricta (< en vez de <=) para evitar ambigüedad
        # en la frontera entre sensores cercanos (ej: SPOT-6/7 1.5m vs Pleiades 0.5m)
        if abs(res - expected_res) < _RES_TOLERANCE_M:
            return sensor_name

    return "desconocido"


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _compute_resolution_m(transform, crs) -> float | None:
    """Calcula la resolución espacial en metros por píxel.

    Para CRS proyectados en metros, usa la componente de escala del transform.
    Para CRS geográficos (grados), retorna None (no convertible sin proyectar).

    Args:
        transform: Objeto :class:`rasterio.transform.Affine`.
        crs: Objeto :class:`rasterio.crs.CRS`.

    Returns:
        Resolución en metros/px, o ``None`` si el CRS no está en metros.
    """
    try:
        from pyproj import CRS as ProjCRS
        proj_crs = ProjCRS.from_user_input(crs)
        axis_info = proj_crs.axis_info
        # Verifica que la unidad lineal sea metros
        if axis_info and axis_info[0].unit_name in ("metre", "meter"):
            # transform.a = tamaño de píxel en X (positivo), transform.e en Y (negativo)
            px_x = abs(transform.a)
            px_y = abs(transform.e)
            return round((px_x + px_y) / 2, 4)
    except Exception:
        pass
    return None
