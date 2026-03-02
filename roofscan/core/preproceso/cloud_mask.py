"""Cloud masking usando la banda SCL de Sentinel-2 L2A.

La banda SCL (Scene Classification Layer) clasifica cada píxel en categorías
que incluyen nubes, sombras, vegetación, suelo desnudo, etc.
Este módulo usa esas clases para generar una máscara binaria y aplicarla
al array de bandas espectrales.

Clases SCL de Sentinel-2 L2A:
    0  No Data
    1  Saturated / Defective
    2  Dark Area Pixels
    3  Cloud Shadows
    4  Vegetation
    5  Not Vegetated (Bare Soils)
    6  Water
    7  Unclassified
    8  Cloud Medium Probability
    9  Cloud High Probability
   10  Thin Cirrus
   11  Snow / Ice

Píxeles válidos para análisis: clases 4, 5, 6, 7, 11 (y opcionalmente 2).
Píxeles problemáticos: 0, 1, 3, 8, 9, 10.

Uso típico::

    from roofscan.core.preproceso.cloud_mask import apply_cloud_mask, compute_cloud_coverage

    # scl_array: np.ndarray 2D con los valores de la banda SCL
    cloud_pct = compute_cloud_coverage(scl_array)
    masked_array, valid_mask = apply_cloud_mask(spectral_array, scl_array)
"""

import logging

import numpy as np

log = logging.getLogger(__name__)

# Clases SCL consideradas nubosas o inválidas para el análisis de techos
SCL_INVALID_CLASSES = frozenset([0, 1, 3, 8, 9, 10])

# Clases SCL válidas para el análisis
SCL_VALID_CLASSES = frozenset([2, 4, 5, 6, 7, 11])

# Descripción legible de cada clase SCL
SCL_CLASS_NAMES = {
    0: "No Data",
    1: "Saturado/Defectuoso",
    2: "Área oscura",
    3: "Sombra de nube",
    4: "Vegetación",
    5: "Suelo desnudo",
    6: "Agua",
    7: "Sin clasificar",
    8: "Nube media probabilidad",
    9: "Nube alta probabilidad",
    10: "Cirrus",
    11: "Nieve/Hielo",
}


def apply_cloud_mask(
    spectral_array: np.ndarray,
    scl_array: np.ndarray,
    nodata_value: float = np.nan,
    invalid_classes: frozenset[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Aplica la máscara de nubes al array espectral.

    Reemplaza los píxeles inválidos (nubosos, sombras, sin dato) con
    ``nodata_value``. Los píxeles válidos se conservan sin cambios.

    Args:
        spectral_array: Array NumPy de forma ``(bandas, alto, ancho)`` con
                        los datos espectrales a enmascarar.
        scl_array: Array NumPy 2D ``(alto, ancho)`` con las clases SCL.
                   Debe tener las mismas dimensiones espaciales que ``spectral_array``.
        nodata_value: Valor a asignar a los píxeles inválidos. Por defecto ``np.nan``
                      (requiere dtype float). Usar 0 para datos enteros.
        invalid_classes: Conjunto de clases SCL a enmascarar. Si es ``None``,
                         usa :data:`SCL_INVALID_CLASSES`.

    Returns:
        Tupla ``(masked_array, valid_mask)`` donde:

        - ``masked_array``: Array float32 de forma ``(bandas, alto, ancho)``
          con nodata en píxeles inválidos.
        - ``valid_mask``: Array booleano 2D ``(alto, ancho)``. ``True``
          indica píxel válido.

    Raises:
        ValueError: Si las dimensiones espaciales de ``spectral_array`` y
                    ``scl_array`` no coinciden.
    """
    if invalid_classes is None:
        invalid_classes = SCL_INVALID_CLASSES

    _validate_dimensions(spectral_array, scl_array)

    # Máscara booleana: True donde el píxel es VÁLIDO
    valid_mask = ~np.isin(scl_array, list(invalid_classes))

    # Convertir a float32 para poder usar NaN como nodata
    masked = spectral_array.astype(np.float32)

    # Aplicar máscara: nodata en píxeles inválidos
    masked[:, ~valid_mask] = nodata_value

    n_total = valid_mask.size
    n_valid = valid_mask.sum()
    pct_valid = 100.0 * n_valid / n_total if n_total > 0 else 0.0
    log.info(
        "Cloud mask aplicada | válidos: %d/%d (%.1f%%)",
        n_valid, n_total, pct_valid,
    )

    return masked, valid_mask


def compute_cloud_coverage(scl_array: np.ndarray) -> float:
    """Calcula el porcentaje de píxeles nubosos en la imagen.

    Considera como nubosos las clases SCL 3 (shadow), 8, 9 (clouds) y 10 (cirrus).
    Los píxeles de No Data (clase 0) se excluyen del denominador.

    Args:
        scl_array: Array NumPy 2D con las clases SCL.

    Returns:
        Porcentaje de nubosidad (0.0–100.0). Retorna 100.0 si todos los
        píxeles son No Data.
    """
    cloudy_classes = frozenset([3, 8, 9, 10])
    total_valid = np.sum(scl_array != 0)
    if total_valid == 0:
        return 100.0
    n_cloudy = np.sum(np.isin(scl_array, list(cloudy_classes)))
    return float(100.0 * n_cloudy / total_valid)


def scl_class_summary(scl_array: np.ndarray) -> dict[str, float]:
    """Calcula el porcentaje de cada clase SCL presente en la imagen.

    Útil para diagnóstico y para informar al usuario sobre la calidad
    de la imagen antes de procesarla.

    Args:
        scl_array: Array NumPy 2D con las clases SCL.

    Returns:
        Dict con el nombre de cada clase y su porcentaje de cobertura.
        Solo incluye las clases presentes (> 0 píxeles).

    Example::

        summary = scl_class_summary(scl_band)
        # {'Vegetación': 45.2, 'Suelo desnudo': 30.1, 'Nube alta probabilidad': 12.3, ...}
    """
    total = scl_array.size
    if total == 0:
        return {}

    result = {}
    unique, counts = np.unique(scl_array, return_counts=True)
    for cls_val, count in zip(unique, counts):
        pct = 100.0 * count / total
        name = SCL_CLASS_NAMES.get(int(cls_val), f"Clase {cls_val}")
        result[name] = round(pct, 2)

    return result


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _validate_dimensions(spectral_array: np.ndarray, scl_array: np.ndarray) -> None:
    """Verifica que las dimensiones espaciales coincidan."""
    if spectral_array.ndim != 3:
        raise ValueError(
            f"spectral_array debe ser 3D (bandas, alto, ancho), "
            f"recibido shape={spectral_array.shape}"
        )
    if scl_array.ndim != 2:
        raise ValueError(
            f"scl_array debe ser 2D (alto, ancho), "
            f"recibido shape={scl_array.shape}"
        )
    _, s_h, s_w = spectral_array.shape
    scl_h, scl_w = scl_array.shape
    if (s_h, s_w) != (scl_h, scl_w):
        raise ValueError(
            f"Las dimensiones espaciales no coinciden: "
            f"spectral=({s_h}, {s_w}) vs scl=({scl_h}, {scl_w}). "
            "Ambos arrays deben tener el mismo alto y ancho."
        )
