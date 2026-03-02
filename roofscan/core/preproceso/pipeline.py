"""Pipeline de preprocesamiento de imágenes satelitales.

Encadena los pasos de reproyección, cloud masking y normalización en un
único punto de entrada, manteniendo cada paso independiente y reutilizable.

El pipeline opera sobre el dict estándar producido por
:func:`~roofscan.core.ingesta.loader.load_geotiff` y añade claves nuevas
al resultado sin modificar las originales.

Uso típico::

    from roofscan.core.ingesta.loader import load_geotiff
    from roofscan.core.preproceso.pipeline import run_preprocessing

    data = load_geotiff("S2_scene/B02.tif")
    scl = load_geotiff("S2_scene/SCL.tif")["array"][0]

    result = run_preprocessing(data, scl_array=scl)
    # result["array"]       → array normalizado (bandas, H, W) float32
    # result["valid_mask"]  → máscara booleana 2D
    # result["cloud_pct"]   → % nubosidad de la escena
    # result["stats"]       → estadísticas por banda post-normalización
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from roofscan.config import CRS_WORK

log = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuración del pipeline de preprocesamiento.

    Attributes:
        target_crs: CRS de destino para la reproyección.
        max_cloud_pct: Porcentaje máximo de nubosidad tolerable. Si la escena
                       supera este valor se emite un aviso pero el procesamiento
                       continúa (no se lanza excepción).
        nodata_value: Valor de nodata en el array de entrada (antes de normalizar).
        s2_scale_factor: Factor de escala DN → reflectancia para Sentinel-2 L2A.
        clip_percentiles: Tupla (low, high) para recorte de outliers. ``None``
                          para desactivar.
        skip_reproject: Si ``True``, omite la reproyección (útil si la imagen
                        ya está en el CRS correcto).
    """
    target_crs: str = CRS_WORK
    max_cloud_pct: float = 20.0
    nodata_value: float = 0.0
    s2_scale_factor: float = 10_000.0
    clip_percentiles: tuple[float, float] | None = (2.0, 98.0)
    skip_reproject: bool = False


def run_preprocessing(
    data: dict[str, Any],
    scl_array: np.ndarray | None = None,
    config: PreprocessConfig | None = None,
) -> dict[str, Any]:
    """Ejecuta el pipeline completo de preprocesamiento sobre un raster.

    Pasos (en orden):

    1. **Reproyección** → CRS de trabajo (EPSG:32720) si corresponde.
    2. **Cloud masking** → usando la banda SCL (si se provee).
    3. **Normalización** → DN de Sentinel-2 a reflectancia [0.0, 1.0].
    4. **Recorte de percentiles** → para visualización (opcional).
    5. **Estadísticas** → por banda del array final.

    Args:
        data: Dict retornado por :func:`~roofscan.core.ingesta.loader.load_geotiff`.
        scl_array: Array NumPy 2D ``(alto, ancho)`` con la banda SCL de
                   Sentinel-2 L2A. Si es ``None``, se omite el cloud masking.
        config: Configuración del pipeline. Si es ``None`` usa los valores
                por defecto de :class:`PreprocessConfig`.

    Returns:
        Dict con todos los campos originales de ``data`` más:

        - ``array``: Array float32 normalizado ``(bandas, H, W)``.
        - ``valid_mask``: Máscara booleana 2D (``True`` = válido). ``None``
          si no se aplicó cloud masking.
        - ``cloud_pct``: Porcentaje de nubosidad. ``None`` si no hay SCL.
        - ``scl_summary``: Resumen de clases SCL. ``None`` si no hay SCL.
        - ``stats``: Lista de dicts con estadísticas por banda.
        - ``preprocessing_config``: Copia de la configuración usada.

    Raises:
        ValueError: Si el array de entrada tiene dimensiones inválidas.
    """
    if config is None:
        config = PreprocessConfig()

    from roofscan.core.preproceso.reprojector import reproject_to_crs
    from roofscan.core.preproceso.cloud_mask import (
        apply_cloud_mask, compute_cloud_coverage, scl_class_summary,
    )
    from roofscan.core.preproceso.normalizer import normalize_s2, clip_percentile, band_statistics

    result = dict(data)  # copia superficial del dict; no modifica el original

    # -- Paso 1: Reproyección --------------------------------------------------
    if not config.skip_reproject:
        log.info("[Pipeline] Paso 1/4: Reproyección → %s", config.target_crs)
        result = reproject_to_crs(result, target_crs=config.target_crs)

        # Si hay SCL, también hay que reproyectarla al mismo CRS
        if scl_array is not None:
            scl_array = _reproject_scl(scl_array, data, result)
    else:
        log.info("[Pipeline] Paso 1/4: Reproyección omitida (skip_reproject=True)")

    # -- Paso 2: Cloud masking ------------------------------------------------
    valid_mask = None
    cloud_pct = None
    summary = None

    if scl_array is not None:
        log.info("[Pipeline] Paso 2/4: Cloud masking")
        cloud_pct = compute_cloud_coverage(scl_array)
        summary = scl_class_summary(scl_array)

        if cloud_pct > config.max_cloud_pct:
            log.warning(
                "Nubosidad elevada: %.1f%% (límite configurado: %.1f%%). "
                "Los resultados pueden ser poco confiables.",
                cloud_pct, config.max_cloud_pct,
            )

        result["array"], valid_mask = apply_cloud_mask(
            result["array"],
            scl_array,
            nodata_value=np.nan,
        )
    else:
        log.info("[Pipeline] Paso 2/4: Cloud masking omitido (sin banda SCL)")

    # -- Paso 3: Normalización ------------------------------------------------
    log.info("[Pipeline] Paso 3/4: Normalización (factor=%.0f)", config.s2_scale_factor)
    result["array"] = normalize_s2(
        result["array"],
        nodata=config.nodata_value,
        scale_factor=config.s2_scale_factor,
    )

    # -- Paso 4: Recorte de percentiles (opcional) ----------------------------
    if config.clip_percentiles is not None:
        lo, hi = config.clip_percentiles
        log.info("[Pipeline] Paso 4/4: Recorte percentiles [%.0f%%, %.0f%%]", lo, hi)
        result["array"] = clip_percentile(result["array"], low_pct=lo, high_pct=hi)
    else:
        log.info("[Pipeline] Paso 4/4: Recorte de percentiles omitido")

    # -- Estadísticas finales -------------------------------------------------
    stats = band_statistics(result["array"])

    result.update({
        "valid_mask": valid_mask,
        "cloud_pct": cloud_pct,
        "scl_summary": summary,
        "stats": stats,
        "preprocessing_config": config,
    })

    log.info(
        "[Pipeline] Preprocesamiento completo | shape=%s | nubosidad=%s",
        result["array"].shape,
        f"{cloud_pct:.1f}%" if cloud_pct is not None else "N/A",
    )
    return result


# ---------------------------------------------------------------------------
# Helper privado
# ---------------------------------------------------------------------------

def _reproject_scl(
    scl_array: np.ndarray,
    original_data: dict[str, Any],
    reprojected_data: dict[str, Any],
) -> np.ndarray:
    """Reproyecta la banda SCL al CRS y tamaño del raster ya reproyectado.

    Usa nearest neighbor para preservar los valores de clase enteros.
    """
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.warp import reproject, Resampling
    except ImportError:
        log.warning("rasterio no disponible; la banda SCL no será reproyectada.")
        return scl_array

    src_crs = CRS.from_user_input(original_data["crs"])
    dst_crs = CRS.from_user_input(reprojected_data["crs"])

    if src_crs == dst_crs:
        return scl_array

    _, dst_h, dst_w = reprojected_data["array"].shape
    dst_scl = np.zeros((dst_h, dst_w), dtype=scl_array.dtype)

    reproject(
        source=scl_array,
        destination=dst_scl,
        src_transform=original_data["transform"],
        src_crs=src_crs,
        dst_transform=reprojected_data["transform"],
        dst_crs=dst_crs,
        resampling=Resampling.nearest,  # nearest neighbor para clases discretas
    )
    return dst_scl
