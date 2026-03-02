"""Vectorización de la máscara de techos a geometrías GeoDataFrame.

Convierte el mapa de etiquetas raster a polígonos georreferenciados,
que pueden exportarse como GeoJSON o Shapefile y visualizarse en QGIS.

Uso típico::

    from roofscan.core.calculo.geometry_merger import labels_to_geodataframe

    gdf = labels_to_geodataframe(labels, transform, crs, areas)
    gdf.to_file("techos.geojson", driver="GeoJSON")
"""

import logging

import numpy as np

log = logging.getLogger(__name__)


def labels_to_geodataframe(
    labels: np.ndarray,
    transform,
    crs: str,
    areas: list[dict] | None = None,
):
    """Vectoriza el mapa de etiquetas a un GeoDataFrame con polígonos.

    Cada región etiquetada se convierte en un polígono georreferenciado.
    Opcionalmente añade los atributos de área (m²) a cada polígono.

    Args:
        labels: Array int32 2D ``(H, W)`` con IDs de objetos (0 = fondo).
        transform: Objeto :class:`rasterio.transform.Affine` de georreferenciación.
        crs: CRS del raster en formato EPSG o WKT (ej. ``"EPSG:32720"``).
        areas: Lista de dicts retornada por :func:`~roofscan.core.calculo.area_calculator.calculate_areas`.
               Si se provee, agrega columnas ``area_m2`` y ``area_px``.

    Returns:
        :class:`geopandas.GeoDataFrame` con columnas:

        - ``roof_id``: ID del objeto.
        - ``geometry``: Polígono en el CRS dado.
        - ``area_m2``: Área en m² (si se provee ``areas``).
        - ``area_px``: Área en píxeles (si se provee ``areas``).

    Raises:
        ImportError: Si rasterio, shapely o geopandas no están instalados.
        ValueError: Si ``labels`` no contiene ningún objeto.
    """
    try:
        import geopandas as gpd
        from rasterio.features import shapes
        from shapely.geometry import shape
    except ImportError as exc:
        raise ImportError(
            "Dependencias faltantes: rasterio, shapely, geopandas. "
            "Ejecutá: pip install rasterio shapely geopandas"
        ) from exc

    if labels.max() == 0:
        raise ValueError(
            "El mapa de etiquetas no contiene objetos detectados (todos son 0). "
            "Verificá la detección antes de vectorizar."
        )

    log.info("Vectorizando %d objetos a GeoDataFrame...", int(labels.max()))

    # rasterio.features.shapes genera (geometry_dict, value) por cada región
    mask = (labels > 0).astype(np.uint8)
    polys = []
    for geom_dict, value in shapes(labels.astype(np.int32), mask=mask, transform=transform):
        if value == 0:
            continue
        polys.append({
            "roof_id": int(value),
            "geometry": shape(geom_dict),
        })

    if not polys:
        raise ValueError("No se generaron polígonos válidos desde el mapa de etiquetas.")

    gdf = gpd.GeoDataFrame(polys, crs=crs)

    # Fusionar atributos de área si se proveen
    if areas:
        area_map = {r["id"]: r for r in areas}
        gdf["area_m2"] = gdf["roof_id"].map(
            lambda rid: area_map.get(rid, {}).get("area_m2", None)
        )
        gdf["area_px"] = gdf["roof_id"].map(
            lambda rid: area_map.get(rid, {}).get("area_px", None)
        )

    log.info("GeoDataFrame creado | %d polígonos | CRS=%s", len(gdf), crs)
    return gdf


def reproject_geodataframe(gdf, target_crs: str):
    """Reproyecta un GeoDataFrame a otro CRS.

    Args:
        gdf: :class:`geopandas.GeoDataFrame` con CRS definido.
        target_crs: CRS de destino (ej. ``"EPSG:4326"`` para WGS84).

    Returns:
        Nuevo GeoDataFrame en el CRS de destino.
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("Instalar geopandas: pip install geopandas") from exc

    if gdf.crs is None:
        raise ValueError("El GeoDataFrame no tiene CRS definido.")

    return gdf.to_crs(target_crs)
