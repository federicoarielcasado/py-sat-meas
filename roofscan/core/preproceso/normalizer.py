"""Normalización radiométrica de imágenes Sentinel-2.

Sentinel-2 L2A entrega valores de reflectancia de superficie en el rango
0–10000 (enteros uint16). Este módulo escala esos valores a float32 en [0.0, 1.0],
lo que facilita el cálculo de índices espectrales y la alimentación de modelos DL.

También incluye utilidades para calcular estadísticas por banda y recortar
valores atípicos (clipping de percentiles) antes de normalizar.

Uso típico::

    from roofscan.core.preproceso.normalizer import normalize_s2, clip_percentile

    # array: np.ndarray float32 (bandas, alto, ancho), ya con cloud mask aplicada
    arr_norm = normalize_s2(array)          # escala de 0-10000 a 0.0-1.0
    arr_clipped = clip_percentile(arr_norm) # recorta outliers antes de visualizar
"""

import logging

import numpy as np

log = logging.getLogger(__name__)

# Factor de escala estándar de Sentinel-2 L2A (DN → reflectancia)
S2_SCALE_FACTOR = 10_000.0

# Reflectancia máxima esperada (algunos píxeles pueden superar 1.0 por ruido)
S2_REFLECTANCE_MAX = 1.0

# Percentiles por defecto para recorte de outliers en visualización
DEFAULT_CLIP_LOW_PCT = 2.0
DEFAULT_CLIP_HIGH_PCT = 98.0


def normalize_s2(
    array: np.ndarray,
    nodata: float | None = None,
    scale_factor: float = S2_SCALE_FACTOR,
) -> np.ndarray:
    """Escala un array Sentinel-2 L2A de DN a reflectancia [0.0, 1.0].

    Divide los valores entre ``scale_factor`` (10 000 por defecto).
    Los valores resultado se recortan a [0.0, 1.0] para eliminar DN
    anómalos. Los píxeles nodata (NaN o valor explícito) se preservan.

    Args:
        array: Array NumPy de forma ``(bandas, alto, ancho)`` o ``(alto, ancho)``
               con valores DN de Sentinel-2 L2A (típicamente uint16 o float32).
        nodata: Valor de no-dato a preservar sin escalar. Si el array ya tiene
                NaN (por cloud mask), pasar ``None``.
        scale_factor: Factor divisor. Por defecto 10 000 (estándar S2 L2A).

    Returns:
        Array float32 de la misma forma que la entrada, con valores en [0.0, 1.0].
        Los píxeles nodata se convierten a ``np.nan``.
    """
    result = array.astype(np.float32)

    if nodata is not None:
        nodata_mask = result == nodata
    else:
        nodata_mask = np.isnan(result)

    # Escalar
    result = result / scale_factor

    # Recortar rango físico válido
    valid = ~nodata_mask
    result[valid] = np.clip(result[valid], 0.0, S2_REFLECTANCE_MAX)

    # Restaurar nodata como NaN
    result[nodata_mask] = np.nan

    log.debug(
        "Normalización S2 | shape=%s | válidos=%.1f%%",
        result.shape,
        100.0 * valid.sum() / valid.size if valid.size > 0 else 0.0,
    )
    return result


def clip_percentile(
    array: np.ndarray,
    low_pct: float = DEFAULT_CLIP_LOW_PCT,
    high_pct: float = DEFAULT_CLIP_HIGH_PCT,
    per_band: bool = True,
) -> np.ndarray:
    """Recorta valores de un array a los percentiles dados (ignorando NaN).

    Útil para mejorar el contraste antes de visualizar o para reducir
    el impacto de outliers en índices espectrales.

    Args:
        array: Array NumPy de forma ``(bandas, alto, ancho)`` o ``(alto, ancho)``.
        low_pct: Percentil inferior de recorte (0–100). Por defecto 2.
        high_pct: Percentil superior de recorte (0–100). Por defecto 98.
        per_band: Si ``True``, calcula los percentiles independientemente
                  por banda. Si ``False``, usa estadísticas globales del array.

    Returns:
        Array float32 con los valores recortados al rango [p_low, p_high].

    Raises:
        ValueError: Si ``low_pct >= high_pct`` o están fuera de [0, 100].
    """
    if not (0 <= low_pct < high_pct <= 100):
        raise ValueError(
            f"Percentiles inválidos: low_pct={low_pct}, high_pct={high_pct}. "
            "Deben cumplir 0 ≤ low_pct < high_pct ≤ 100."
        )

    result = array.astype(np.float32)

    if result.ndim == 2 or not per_band:
        # Cálculo global
        valid_vals = result[~np.isnan(result)]
        if valid_vals.size > 0:
            lo, hi = np.nanpercentile(result, [low_pct, high_pct])
            result = np.clip(result, lo, hi)
    else:
        # Por banda
        for i in range(result.shape[0]):
            band = result[i]
            valid_vals = band[~np.isnan(band)]
            if valid_vals.size > 0:
                lo, hi = np.nanpercentile(band, [low_pct, high_pct])
                result[i] = np.clip(band, lo, hi)

    return result


def band_statistics(array: np.ndarray) -> list[dict[str, float]]:
    """Calcula estadísticas básicas por banda (ignorando NaN).

    Args:
        array: Array NumPy de forma ``(bandas, alto, ancho)``.

    Returns:
        Lista de dicts, uno por banda, con claves:
        ``min``, ``max``, ``mean``, ``std``, ``valid_pct``.
    """
    stats = []
    if array.ndim == 2:
        array = array[np.newaxis, ...]

    for i in range(array.shape[0]):
        band = array[i].astype(np.float64)
        valid = band[~np.isnan(band)]
        total = band.size
        if valid.size == 0:
            stats.append({"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan, "valid_pct": 0.0})
        else:
            stats.append({
                "min": float(valid.min()),
                "max": float(valid.max()),
                "mean": float(valid.mean()),
                "std": float(valid.std()),
                "valid_pct": round(100.0 * valid.size / total, 2),
            })
    return stats
