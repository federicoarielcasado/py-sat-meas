"""Reproyección de rasters al CRS de trabajo del proyecto (EPSG:32720 por defecto).

Convierte cualquier raster al sistema de coordenadas proyectado configurado,
necesario para que el cálculo de áreas sea correcto (unidades métricas).

Uso típico::

    from roofscan.core.ingesta.loader import load_geotiff
    from roofscan.core.preproceso.reprojector import reproject_to_crs

    data = load_geotiff("imagen.tif")
    data_utm = reproject_to_crs(data)
    print(data_utm["crs"])  # "EPSG:32720"
"""

import logging
from typing import Any

import numpy as np

from roofscan.config import CRS_WORK

log = logging.getLogger(__name__)


def reproject_to_crs(data: dict[str, Any], target_crs: str = CRS_WORK) -> dict[str, Any]:
    """Reproyecta un raster al CRS de destino.

    Si el raster ya está en el CRS solicitado, lo retorna sin modificaciones
    (sin copiar el array para no desperdiciar memoria).

    Args:
        data: Dict retornado por :func:`~roofscan.core.ingesta.loader.load_geotiff`.
              Debe contener al menos ``array``, ``crs``, ``transform`` y ``nodata``.
        target_crs: CRS de destino en formato EPSG o WKT. Por defecto ``CRS_WORK``
                    (``"EPSG:32720"``).

    Returns:
        Nuevo dict con la misma estructura que ``data`` pero con ``array``,
        ``crs``, ``transform``, ``bounds`` y ``resolution_m`` actualizados.

    Raises:
        ImportError: Si rasterio o pyproj no están instalados.
        ValueError: Si el CRS de origen no está definido en ``data``.
    """
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.warp import calculate_default_transform, reproject, Resampling
        from rasterio.transform import array_bounds
    except ImportError as exc:
        raise ImportError(
            "La librería 'rasterio' no está instalada. Ejecutá: pip install rasterio"
        ) from exc

    src_crs_str = data.get("crs")
    if not src_crs_str:
        raise ValueError("El dict de entrada no tiene CRS definido ('crs' es None o vacío).")

    src_crs = CRS.from_user_input(src_crs_str)
    dst_crs = CRS.from_user_input(target_crs)

    # Normalizar para comparar (mismo CRS → no reproyectar)
    if src_crs == dst_crs:
        log.debug("El raster ya está en %s, no se reproyecta.", target_crs)
        return data

    array = data["array"]          # (bandas, alto, ancho)
    src_transform = data["transform"]
    nodata = data.get("nodata", 0)
    n_bands, src_height, src_width = array.shape

    log.info("Reproyectando %s → %s ...", src_crs_str, target_crs)

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, src_width, src_height, *_bounds_from_data(data)
    )

    dst_array = np.full(
        (n_bands, dst_height, dst_width),
        fill_value=nodata if nodata is not None else 0,
        dtype=array.dtype,
    )

    for band_idx in range(n_bands):
        reproject(
            source=array[band_idx],
            destination=dst_array[band_idx],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=nodata,
            dst_nodata=nodata,
        )

    dst_bounds = array_bounds(dst_height, dst_width, dst_transform)
    resolution_m = _compute_resolution_m(dst_transform, dst_crs)

    log.info(
        "Reproyección completa | nuevo tamaño=(%d, %d) | res=%.1f m/px",
        dst_height, dst_width, resolution_m or 0,
    )

    return {
        **data,
        "array": dst_array,
        "crs": dst_crs.to_string(),
        "transform": dst_transform,
        "bounds": dst_bounds,
        "resolution_m": resolution_m,
    }


def is_metric_crs(crs_str: str) -> bool:
    """Devuelve ``True`` si el CRS dado usa metros como unidad lineal.

    Args:
        crs_str: CRS en formato EPSG, WKT o proj4.

    Returns:
        ``True`` si la unidad es metros, ``False`` en caso contrario.
    """
    try:
        from pyproj import CRS as ProjCRS
        crs = ProjCRS.from_user_input(crs_str)
        axis_info = crs.axis_info
        return bool(axis_info and axis_info[0].unit_name in ("metre", "meter"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _bounds_from_data(data: dict[str, Any]) -> tuple[float, float, float, float]:
    """Extrae (left, bottom, right, top) del dict de datos de raster."""
    bounds = data.get("bounds")
    if bounds is not None:
        return bounds.left, bounds.bottom, bounds.right, bounds.top
    # Calcular desde transform si no hay bounds
    from rasterio.transform import array_bounds
    _, h, w = data["array"].shape
    b = array_bounds(h, w, data["transform"])
    return b.left, b.bottom, b.right, b.top


def _compute_resolution_m(transform, crs) -> float | None:
    """Calcula la resolución en m/px si el CRS es métrico."""
    try:
        from pyproj import CRS as ProjCRS
        proj_crs = ProjCRS.from_user_input(crs)
        if proj_crs.axis_info and proj_crs.axis_info[0].unit_name in ("metre", "meter"):
            return round((abs(transform.a) + abs(transform.e)) / 2, 4)
    except Exception:
        pass
    return None
