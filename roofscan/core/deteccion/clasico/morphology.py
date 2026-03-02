"""Operaciones morfológicas para limpiar y refinar la máscara de detección.

Después de aplicar los índices espectrales, la máscara binaria contiene:
- Ruido de píxeles aislados (falsos positivos pequeños)
- Agujeros dentro de techos reales
- Bordes irregulares

Este módulo aplica operaciones morfológicas estándar para producir
polígonos limpios y realistas.

Usa scipy.ndimage como motor primario (siempre disponible) y opcionalmente
opencv-python-headless para kernels más flexibles.

Uso típico::

    from roofscan.core.deteccion.clasico.morphology import clean_mask, label_roofs

    clean = clean_mask(raw_mask, min_area_px=5, close_radius=1)
    labels, n_roofs = label_roofs(clean)
    # labels: array 2D con int, cada techo tiene un ID único
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

log = logging.getLogger(__name__)

# Elemento estructurante cuadrado 3×3 (conectividad 8 vecinos)
_SE_3x3 = np.ones((3, 3), dtype=bool)


@dataclass
class MorphologyConfig:
    """Parámetros de limpieza morfológica.

    Attributes:
        min_area_px: Tamaño mínimo de objeto en píxeles. Objetos más pequeños
                     se eliminan como ruido. Para Sentinel-2 a 10 m/px, 5 px ≈ 500 m².
        close_radius: Radio del elemento estructurante para cierre morfológico
                      (une huecos y conecta objetos cercanos). En píxeles.
        open_radius: Radio para apertura morfológica (elimina ruido fino). En píxeles.
        fill_holes: Si ``True``, rellena agujeros internos en los objetos detectados.
    """
    min_area_px: int = 5
    close_radius: int = 1
    open_radius: int = 1
    fill_holes: bool = True


def clean_mask(
    mask: np.ndarray,
    config: MorphologyConfig | None = None,
) -> np.ndarray:
    """Aplica operaciones morfológicas para limpiar una máscara binaria.

    Secuencia de pasos:
    1. Apertura (elimina ruido fino)
    2. Cierre (conecta objetos próximos, rellena huecos pequeños)
    3. Relleno de agujeros internos (opcional)
    4. Eliminación de objetos pequeños

    Args:
        mask: Array booleano 2D ``(H, W)``. ``True`` = píxel detectado como techo.
        config: Parámetros de limpieza. Si es ``None`` usa :class:`MorphologyConfig`
                por defecto.

    Returns:
        Array booleano 2D limpio con la misma forma que ``mask``.

    Raises:
        ValueError: Si ``mask`` no es 2D.
    """
    if config is None:
        config = MorphologyConfig()

    if mask.ndim != 2:
        raise ValueError(
            f"La máscara debe ser 2D (H, W), recibido shape={mask.shape}"
        )

    result = mask.astype(bool)
    n_before = result.sum()

    # Paso 1: Apertura (erosión + dilatación) — elimina ruido fino
    if config.open_radius > 0:
        se = _disk_se(config.open_radius)
        result = ndi.binary_opening(result, structure=se)

    # Paso 2: Cierre (dilatación + erosión) — conecta objetos y rellena pequeños huecos
    if config.close_radius > 0:
        se = _disk_se(config.close_radius)
        result = ndi.binary_closing(result, structure=se)

    # Paso 3: Relleno de agujeros internos
    if config.fill_holes:
        result = ndi.binary_fill_holes(result)

    # Paso 4: Eliminar objetos pequeños por área mínima
    if config.min_area_px > 1:
        result = _remove_small_objects(result, config.min_area_px)

    n_after = result.sum()
    log.info(
        "Morfología | px antes=%d, después=%d (Δ=%+d)",
        n_before, n_after, n_after - n_before,
    )
    return result


def label_roofs(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Etiqueta objetos conectados en la máscara limpia.

    Asigna un entero único a cada región conectada (techo individual).
    Usa conectividad de 8 vecinos.

    Args:
        mask: Array booleano 2D ``(H, W)`` limpio.

    Returns:
        Tupla ``(labeled_array, n_objects)`` donde:

        - ``labeled_array``: Array int32 2D. Cada techo tiene un ID >= 1.
          El fondo es 0.
        - ``n_objects``: Número de techos detectados.
    """
    labeled, n = ndi.label(mask, structure=_SE_3x3)
    log.info("Etiquetado: %d techos detectados", n)
    return labeled.astype(np.int32), n


def run_morphology(
    mask: np.ndarray,
    config: MorphologyConfig | None = None,
) -> dict:
    """Pipeline morfológico completo: limpieza + etiquetado.

    Args:
        mask: Array booleano 2D de detección cruda (salida de ``detect_roofs``).
        config: Parámetros de morfología.

    Returns:
        Dict con:

        - ``mask_clean`` (np.ndarray bool 2D): Máscara limpia.
        - ``labels`` (np.ndarray int32 2D): Mapa de etiquetas.
        - ``n_roofs`` (int): Número de techos detectados.
        - ``morphology_config``: Configuración usada.
    """
    if config is None:
        config = MorphologyConfig()

    mask_clean = clean_mask(mask, config)
    labels, n_roofs = label_roofs(mask_clean)

    return {
        "mask_clean": mask_clean,
        "labels": labels,
        "n_roofs": n_roofs,
        "morphology_config": config,
    }


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _disk_se(radius: int) -> np.ndarray:
    """Crea un elemento estructurante circular de radio dado."""
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x ** 2 + y ** 2 <= radius ** 2).astype(bool)


def _remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Elimina regiones conectadas con área < min_size píxeles."""
    labeled, _ = ndi.label(mask, structure=_SE_3x3)
    sizes = ndi.sum(mask, labeled, range(1, labeled.max() + 1))
    # Construir máscara de objetos grandes
    large_labels = np.where(np.array(sizes) >= min_size)[0] + 1
    result = np.isin(labeled, large_labels)
    return result.astype(bool)
