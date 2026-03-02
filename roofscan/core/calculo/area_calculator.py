"""Cálculo de área de techos detectados en metros cuadrados.

Convierte la cantidad de píxeles de cada región etiquetada a m²,
usando la resolución espacial del raster (metros/píxel).

Uso típico::

    from roofscan.core.calculo.area_calculator import calculate_areas

    areas = calculate_areas(labels, resolution_m=10.0)
    # [{"id": 1, "area_m2": 250.0, "area_px": 3}, ...]
"""

import logging

import numpy as np
from scipy import ndimage as ndi

log = logging.getLogger(__name__)


def calculate_areas(
    labels: np.ndarray,
    resolution_m: float,
    min_area_m2: float = 0.0,
) -> list[dict]:
    """Calcula el área en m² de cada región etiquetada.

    Args:
        labels: Array int32 2D ``(H, W)`` con regiones numeradas desde 1.
                El valor 0 corresponde al fondo (ignorado).
        resolution_m: Resolución espacial en metros/píxel. Para Sentinel-2, 10.0.
        min_area_m2: Área mínima en m² para incluir en los resultados.
                     Objetos más pequeños se filtran del output pero no de
                     ``labels`` (el filtrado morfológico ya ocurrió antes).

    Returns:
        Lista de dicts ordenada por área descendente. Cada dict contiene:

        - ``id`` (int): ID del objeto (valor en ``labels``).
        - ``area_px`` (int): Área en píxeles.
        - ``area_m2`` (float): Área en metros cuadrados.
        - ``centroid_px`` (tuple[float, float]): Centroide ``(fila, columna)`` en píxeles.

    Raises:
        ValueError: Si ``resolution_m`` es menor o igual a 0.
    """
    if resolution_m <= 0:
        raise ValueError(
            f"resolution_m debe ser mayor que 0, recibido: {resolution_m}"
        )

    pixel_area_m2 = resolution_m ** 2
    n_objects = int(labels.max())

    if n_objects == 0:
        log.info("No se detectaron objetos para calcular área.")
        return []

    label_ids = range(1, n_objects + 1)

    # Cantidad de píxeles por objeto
    areas_px = ndi.sum(labels > 0, labels, label_ids)

    # Centroides (fila, columna) en coordenadas de píxel
    centroids = ndi.center_of_mass(labels > 0, labels, label_ids)

    results = []
    for label_id, area_px, centroid in zip(label_ids, areas_px, centroids):
        area_m2 = float(area_px) * pixel_area_m2
        if area_m2 < min_area_m2:
            continue
        results.append({
            "id": int(label_id),
            "area_px": int(area_px),
            "area_m2": round(area_m2, 2),
            "centroid_px": (round(centroid[0], 2), round(centroid[1], 2)),
        })

    results.sort(key=lambda x: x["area_m2"], reverse=True)
    total_m2 = sum(r["area_m2"] for r in results)
    log.info(
        "Áreas calculadas | %d objetos | área total=%.1f m²",
        len(results), total_m2,
    )
    return results


def total_covered_area_m2(areas: list[dict]) -> float:
    """Suma el área total cubierta (todos los objetos detectados).

    Args:
        areas: Lista retornada por :func:`calculate_areas`.

    Returns:
        Área total en metros cuadrados.
    """
    return round(sum(r["area_m2"] for r in areas), 2)
