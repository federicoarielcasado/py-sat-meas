"""Clasificación de tipo de estructura a partir de atributos geométricos.

Etapa A de la clasificación de estructuras: usa únicamente la geometría
(área, compacidad, elongación) para asignar una categoría orientativa.
No requiere imagen satelital ni modelo de ML.

Las reglas están calibradas para el tejido urbano de Luján, Buenos Aires:
  - Viviendas residenciales: 40–400 m², forma compacta.
  - Galpones / naves: ≥ 300 m², forma elongada o muy compacta pero grande.
  - Industrial / comercial: ≥ 1500 m².

Dos niveles de clasificación:

1. **Polígono de techo** (``classify_by_geometry``): opera sobre el GeoDataFrame
   de detección individual (salida de ``geometry_merger.labels_to_geodataframe``).
   Usa compacidad y elongación del polígono.

2. **Resumen por parcela** (``classify_parcela``): opera sobre el GeoDataFrame
   de mensura masiva (salida de ``spatial_join.join_roofs_to_parcelas``).
   Usa área total de techos y porcentaje de cobertura.

Uso típico::

    from roofscan.core.calculo.classifier import classify_by_geometry, classify_parcela

    # Clasificar techos individuales
    gdf_roofs = classify_by_geometry(gdf_roofs)

    # Clasificar resumen por parcela (output de batch_mensura)
    resultado = classify_parcela(resultado)
"""

from __future__ import annotations

import logging
import math

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Etiquetas canónicas
# ---------------------------------------------------------------------------

LABEL_VIVIENDA = "vivienda"
LABEL_GALPON = "galpon_nave"
LABEL_INDUSTRIAL = "industrial"
LABEL_OTRO = "otro"

# ---------------------------------------------------------------------------
# Umbrales para clasificación a nivel de polígono de techo
# ---------------------------------------------------------------------------

_AREA_VIVIENDA_MAX_M2: float = 400.0
"""Techos ≤ este valor y compactos → vivienda."""

_AREA_GALPON_MIN_M2: float = 300.0
"""Techos ≥ este valor y elongados → galpón/nave."""

_AREA_INDUSTRIAL_MIN_M2: float = 1500.0
"""Techos ≥ este valor → industrial/comercial, sin importar la forma."""

_COMPACTNESS_THRESHOLD: float = 0.45
"""Compacidad mínima para considerar la forma "compacta" (1 = círculo)."""

_ELONGATION_THRESHOLD: float = 2.0
"""Relación largo/ancho del bounding box mínimo; > umbral → forma elongada."""

# ---------------------------------------------------------------------------
# Umbrales para clasificación a nivel de parcela
# ---------------------------------------------------------------------------

_AREA_PARCELA_INDUSTRIAL_MIN_M2: float = 1500.0
_AREA_PARCELA_GALPON_MIN_M2: float = 300.0
_PCT_GALPON_MAX: float = 70.0
"""Parcelas con alta cobertura y área grande suelen ser industriales/comerciales."""


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def classify_by_geometry(
    gdf: "gpd.GeoDataFrame",
    area_col: str = "area_m2",
    out_col: str = "tipo_estructura",
) -> "gpd.GeoDataFrame":
    """Clasifica polígonos de techo según su forma y área.

    Agrega la columna ``tipo_estructura`` al GeoDataFrame con una de estas
    etiquetas: ``"vivienda"``, ``"galpon_nave"``, ``"industrial"``, ``"otro"``.

    La clasificación es orientativa y está calibrada para el tejido urbano
    de Luján. Para mayor precisión, usar la Etapa B (firma espectral).

    Args:
        gdf: GeoDataFrame con polígonos de techos individuales. Debe contener
            las columnas ``area_m2`` y ``geometry``.
        area_col: Nombre de la columna con el área en m². Default: ``"area_m2"``.
        out_col: Nombre de la columna de salida. Default: ``"tipo_estructura"``.

    Returns:
        Copia del GeoDataFrame con la columna ``out_col`` agregada.

    Raises:
        KeyError: Si no existe la columna ``area_col``.
        ImportError: Si geopandas no está instalado.
    """
    try:
        import geopandas as gpd  # noqa: F401
    except ImportError as exc:
        raise ImportError("geopandas es necesario para classify_by_geometry.") from exc

    if area_col not in gdf.columns:
        raise KeyError(
            f"Columna '{area_col}' no encontrada. "
            f"Columnas disponibles: {list(gdf.columns)}"
        )

    result = gdf.copy()
    result[out_col] = [
        _classify_roof_polygon(row[area_col], row["geometry"])
        for _, row in result.iterrows()
    ]

    counts = result[out_col].value_counts().to_dict()
    log.info("classify_by_geometry | n=%d | distribución: %s", len(result), counts)

    return result


def classify_parcela(
    gdf: "gpd.GeoDataFrame",
    area_col: str = "area_techos_m2",
    pct_col: str = "pct_cubierto",
    out_col: str = "tipo_predominante",
) -> "gpd.GeoDataFrame":
    """Clasifica el tipo predominante de estructura por parcela.

    Opera sobre el GeoDataFrame de mensura masiva (salida de
    ``spatial_join.join_roofs_to_parcelas``), usando el área total de techos
    y el porcentaje de cobertura como señales de clasificación.

    Args:
        gdf: GeoDataFrame con una fila por parcela. Debe contener
            ``area_techos_m2`` y ``pct_cubierto``.
        area_col: Columna con área de techos en m². Default: ``"area_techos_m2"``.
        pct_col: Columna con porcentaje de cobertura. Default: ``"pct_cubierto"``.
        out_col: Nombre de la columna de salida. Default: ``"tipo_predominante"``.

    Returns:
        Copia del GeoDataFrame con la columna ``out_col`` agregada.

    Raises:
        KeyError: Si no existen las columnas requeridas.
    """
    for col in (area_col, pct_col):
        if col not in gdf.columns:
            raise KeyError(
                f"Columna '{col}' no encontrada. "
                f"Columnas disponibles: {list(gdf.columns)}"
            )

    result = gdf.copy()
    result[out_col] = [
        _classify_parcela_row(row[area_col], row[pct_col])
        for _, row in result.iterrows()
    ]

    counts = result[out_col].value_counts().to_dict()
    log.info("classify_parcela | n=%d | distribución: %s", len(result), counts)

    return result


def compute_shape_metrics(geometry) -> dict[str, float]:
    """Calcula métricas de forma de un polígono Shapely.

    Args:
        geometry: Objeto Shapely ``Polygon`` o ``MultiPolygon``.

    Returns:
        Diccionario con:
            - ``area``: área en unidades del CRS.
            - ``perimeter``: perímetro en unidades del CRS.
            - ``compactness``: ``4π·area / perimeter²``
              (1 = círculo perfecto, → 0 = muy elongado o irregular).
            - ``elongation``: relación largo/ancho del bounding box mínimo rotado
              (1 = cuadrado, > 2 = elongado).
    """
    area = geometry.area
    perimeter = geometry.length
    compactness = (4 * math.pi * area / perimeter ** 2) if perimeter > 0 else 0.0

    try:
        mbr = geometry.minimum_rotated_rectangle
        coords = list(mbr.exterior.coords)
        if len(coords) >= 5:
            sides = [math.dist(coords[i], coords[i + 1]) for i in range(4)]
            long_side = max(sides)
            short_side = min(sides)
            elongation = long_side / short_side if short_side > 0 else 1.0
        else:
            elongation = 1.0
    except Exception:
        elongation = 1.0

    return {
        "area": area,
        "perimeter": perimeter,
        "compactness": compactness,
        "elongation": elongation,
    }


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _classify_roof_polygon(area_m2: float, geometry) -> str:
    """Clasifica un polígono de techo individual."""
    if geometry is None or geometry.is_empty:
        return LABEL_OTRO

    metrics = compute_shape_metrics(geometry)
    compactness = metrics["compactness"]
    elongation = metrics["elongation"]
    is_elongated = (
        compactness < _COMPACTNESS_THRESHOLD
        or elongation > _ELONGATION_THRESHOLD
    )

    if area_m2 >= _AREA_INDUSTRIAL_MIN_M2:
        return LABEL_INDUSTRIAL
    if area_m2 >= _AREA_GALPON_MIN_M2 and is_elongated:
        return LABEL_GALPON
    if area_m2 < _AREA_VIVIENDA_MAX_M2 and not is_elongated:
        return LABEL_VIVIENDA
    return LABEL_OTRO


def _classify_parcela_row(area_techos_m2: float, pct_cubierto: float) -> str:
    """Clasifica una fila del resumen por parcela."""
    if area_techos_m2 <= 0:
        return LABEL_OTRO
    if area_techos_m2 >= _AREA_PARCELA_INDUSTRIAL_MIN_M2:
        return LABEL_INDUSTRIAL
    if area_techos_m2 >= _AREA_PARCELA_GALPON_MIN_M2 and pct_cubierto <= _PCT_GALPON_MAX:
        return LABEL_GALPON
    if area_techos_m2 < _AREA_VIVIENDA_MAX_M2:
        return LABEL_VIVIENDA
    return LABEL_OTRO
