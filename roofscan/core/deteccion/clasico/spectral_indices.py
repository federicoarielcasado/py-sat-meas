"""Cálculo de índices espectrales y detección clásica de techos/superficies cubiertas.

Los índices espectrales de Sentinel-2 permiten distinguir superficies
artificiales (techos, pavimento) de vegetación, agua y suelo desnudo.

Convención de bandas para un array de 6 bandas [B02, B03, B04, B08, B11, B12]:
    Índice 0 → B02 (Azul,  ~490 nm)
    Índice 1 → B03 (Verde, ~560 nm)
    Índice 2 → B04 (Rojo,  ~665 nm)
    Índice 3 → B08 (NIR,   ~842 nm)
    Índice 4 → B11 (SWIR1, ~1610 nm)
    Índice 5 → B12 (SWIR2, ~2190 nm)

Índices usados:
    NDVI  = (NIR - Red) / (NIR + Red)    — bajo valor → no vegetación
    NDBI  = (SWIR1 - NIR) / (SWIR1 + NIR) — alto valor → área construida
    NDWI  = (Green - NIR) / (Green + NIR)  — bajo valor → no agua

Estrategia de detección:
    máscara_techo = (NDVI < umbral_ndvi) & (NDBI > umbral_ndbi) & (NDWI < umbral_ndwi)

Uso típico::

    from roofscan.core.deteccion.clasico.spectral_indices import detect_roofs

    result = detect_roofs(normalized_array)
    # result["mask"]  → np.ndarray bool 2D (H, W)
    # result["ndvi"]  → np.ndarray float32 2D
    # result["ndbi"]  → np.ndarray float32 2D
"""

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# Índices de banda (0-based) para el array estándar de 6 bandas S2
_IDX = {
    "B02": 0,  # Azul
    "B03": 1,  # Verde
    "B04": 2,  # Rojo
    "B08": 3,  # NIR
    "B11": 4,  # SWIR1
    "B12": 5,  # SWIR2
}

# Umbrales por defecto calibrados para Luján (techos de zinc, tejas, hormigón)
DEFAULT_NDVI_MAX = 0.20   # por encima → vegetación → no techo
DEFAULT_NDBI_MIN = -0.05  # por debajo → no construido → no techo
DEFAULT_NDWI_MAX = 0.05   # por encima → agua → no techo


@dataclass
class DetectionConfig:
    """Parámetros de detección clásica por índices espectrales.

    Attributes:
        ndvi_max: Umbral superior de NDVI. Píxeles con NDVI > ndvi_max son vegetación.
        ndbi_min: Umbral inferior de NDBI. Píxeles con NDBI < ndbi_min no son construidos.
        ndwi_max: Umbral superior de NDWI. Píxeles con NDWI > ndwi_max son agua.
        band_indices: Dict con los índices 0-based de cada banda en el array.
                      Permite usar arrays con diferente orden de bandas.
    """
    ndvi_max: float = DEFAULT_NDVI_MAX
    ndbi_min: float = DEFAULT_NDBI_MIN
    ndwi_max: float = DEFAULT_NDWI_MAX
    band_indices: dict[str, int] | None = None  # None → usar _IDX estándar


def compute_ndvi(array: np.ndarray, band_indices: dict[str, int] | None = None) -> np.ndarray:
    """Calcula el NDVI (Normalized Difference Vegetation Index).

    NDVI = (NIR − Red) / (NIR + Red)

    Rango: [-1, 1]. Vegetación densa > 0.4; suelo ~0.1; techos/pavimento < 0.2.

    Args:
        array: Array float32 normalizado ``(bandas, H, W)``.
        band_indices: Dict con índices de banda. Usa estándar si es ``None``.

    Returns:
        Array float32 2D ``(H, W)`` con valores en [-1, 1]. NaN donde hay nodata.
    """
    idx = band_indices or _IDX
    nir = array[idx["B08"]].astype(np.float32)
    red = array[idx["B04"]].astype(np.float32)
    return _normalized_difference(nir, red)


def compute_ndbi(array: np.ndarray, band_indices: dict[str, int] | None = None) -> np.ndarray:
    """Calcula el NDBI (Normalized Difference Built-up Index).

    NDBI = (SWIR1 − NIR) / (SWIR1 + NIR)

    Rango: [-1, 1]. Área construida > 0; vegetación < 0.

    Args:
        array: Array float32 normalizado ``(bandas, H, W)``.
        band_indices: Dict con índices de banda.

    Returns:
        Array float32 2D ``(H, W)`` con valores en [-1, 1].
    """
    idx = band_indices or _IDX
    swir = array[idx["B11"]].astype(np.float32)
    nir = array[idx["B08"]].astype(np.float32)
    return _normalized_difference(swir, nir)


def compute_ndwi(array: np.ndarray, band_indices: dict[str, int] | None = None) -> np.ndarray:
    """Calcula el NDWI (Normalized Difference Water Index).

    NDWI = (Green − NIR) / (Green + NIR)

    Rango: [-1, 1]. Agua > 0.3; suelo seco < 0; edificios -0.1 a 0.2.

    Args:
        array: Array float32 normalizado ``(bandas, H, W)``.
        band_indices: Dict con índices de banda.

    Returns:
        Array float32 2D ``(H, W)`` con valores en [-1, 1].
    """
    idx = band_indices or _IDX
    green = array[idx["B03"]].astype(np.float32)
    nir = array[idx["B08"]].astype(np.float32)
    return _normalized_difference(green, nir)


def detect_roofs(
    array: np.ndarray,
    config: DetectionConfig | None = None,
) -> dict:
    """Detecta superficies cubiertas (techos) usando índices espectrales.

    Combina NDVI, NDBI y NDWI con umbrales para generar una máscara binaria
    de posibles techos o superficies impermeables.

    Args:
        array: Array float32 normalizado ``(bandas, H, W)``, salida del pipeline
               de preprocesamiento. Se esperan al menos 6 bandas en orden
               [B02, B03, B04, B08, B11, B12].
        config: Parámetros de detección. Si es ``None`` usa :class:`DetectionConfig`
                con valores por defecto ajustados para Luján.

    Returns:
        Dict con:

        - ``mask`` (np.ndarray bool 2D): ``True`` donde se detecta techo.
        - ``ndvi``, ``ndbi``, ``ndwi`` (np.ndarray float32 2D): índices calculados.
        - ``detection_config``: Configuración usada.
        - ``coverage_pct`` (float): Porcentaje de píxeles detectados como techo.

    Raises:
        ValueError: Si el array tiene menos de 6 bandas o no es float.
    """
    if config is None:
        config = DetectionConfig()

    _validate_array(array)
    idx = config.band_indices or _IDX

    log.info(
        "Detección por índices | NDVI<%.2f, NDBI>%.2f, NDWI<%.2f",
        config.ndvi_max, config.ndbi_min, config.ndwi_max,
    )

    ndvi = compute_ndvi(array, idx)
    ndbi = compute_ndbi(array, idx)
    ndwi = compute_ndwi(array, idx)

    # Máscara de píxeles válidos (sin NaN en ninguno de los índices)
    valid = ~(np.isnan(ndvi) | np.isnan(ndbi) | np.isnan(ndwi))

    # Condiciones de techo
    is_not_vegetation = ndvi < config.ndvi_max
    is_built_up = ndbi > config.ndbi_min
    is_not_water = ndwi < config.ndwi_max

    mask = valid & is_not_vegetation & is_built_up & is_not_water

    n_valid = valid.sum()
    n_detected = mask.sum()
    coverage = float(100.0 * n_detected / n_valid) if n_valid > 0 else 0.0

    log.info(
        "Detección completada | techos: %d px (%.1f%% del área válida)",
        n_detected, coverage,
    )

    return {
        "mask": mask,
        "ndvi": ndvi,
        "ndbi": ndbi,
        "ndwi": ndwi,
        "detection_config": config,
        "coverage_pct": coverage,
    }


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _normalized_difference(band_a: np.ndarray, band_b: np.ndarray) -> np.ndarray:
    """Calcula (A - B) / (A + B) con protección contra división por cero y NaN."""
    denominator = band_a + band_b
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(
            denominator == 0,
            np.nan,
            (band_a - band_b) / denominator,
        ).astype(np.float32)
    # Propagar NaN de las bandas de entrada
    result[np.isnan(band_a) | np.isnan(band_b)] = np.nan
    return result


def _validate_array(array: np.ndarray) -> None:
    """Valida que el array tenga el formato esperado."""
    if array.ndim != 3:
        raise ValueError(
            f"El array debe ser 3D (bandas, H, W), recibido shape={array.shape}"
        )
    if array.shape[0] < 6:
        raise ValueError(
            f"Se necesitan al menos 6 bandas [B02,B03,B04,B08,B11,B12], "
            f"recibidas: {array.shape[0]}. "
            "Verificá que el archivo contiene todas las bandas requeridas."
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(
            f"El array debe ser de tipo float (float32/float64), recibido: {array.dtype}. "
            "Aplicá normalize_s2() antes de detectar."
        )
