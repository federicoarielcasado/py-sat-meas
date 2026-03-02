"""Exportación de resultados de detección a CSV.

Genera un archivo CSV con una fila por objeto detectado, incluyendo área,
centroide en coordenadas proyectadas (metros, EPSG:32720) y en WGS84 si
se provee el GeoDataFrame de geometrías.

Uso típico::

    from roofscan.core.exportacion.csv_exporter import export_csv

    path = export_csv(areas, output_dir="salida/", gdf=gdf_techos)
    print(f"CSV guardado en: {path}")
"""

import csv
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_FIELDS_BASE = ["id", "area_m2", "area_px", "centroid_row", "centroid_col"]
_FIELDS_GEO  = ["centroid_x_m", "centroid_y_m", "centroid_lon", "centroid_lat"]


def export_csv(
    areas: list[dict],
    output_dir: str | Path,
    filename: str = "techos.csv",
    gdf=None,
) -> Path:
    """Exporta la lista de objetos detectados a un archivo CSV.

    Columnas siempre presentes:

    - ``id``: identificador numérico del objeto.
    - ``area_m2``: área en metros cuadrados.
    - ``area_px``: área en píxeles.
    - ``centroid_row``, ``centroid_col``: centroide en coordenadas de píxel.

    Columnas adicionales si ``gdf`` no es ``None``:

    - ``centroid_x_m``, ``centroid_y_m``: centroide en CRS proyectado (metros).
    - ``centroid_lon``, ``centroid_lat``: centroide en WGS84 (grados).

    Args:
        areas: Lista de dicts producida por :func:`~roofscan.core.calculo.area_calculator.calculate_areas`.
        output_dir: Directorio donde guardar el CSV.
        filename: Nombre del archivo (incluir ``.csv``).
        gdf: GeoDataFrame producido por
             :func:`~roofscan.core.calculo.geometry_merger.labels_to_geodataframe`.
             Si se provee, se añaden columnas de centroide georreferenciado.

    Returns:
        :class:`pathlib.Path` del archivo creado.

    Raises:
        ValueError: Si ``areas`` está vacío.
    """
    if not areas:
        raise ValueError("La lista de áreas está vacía; no hay objetos para exportar.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    # Preparar transformación WGS84 si hay GeoDataFrame disponible
    crs_transformer = None
    if gdf is not None and len(gdf) > 0:
        try:
            from pyproj import Transformer
            crs_str = str(gdf.crs)
            if "4326" not in crs_str:
                crs_transformer = Transformer.from_crs(crs_str, "EPSG:4326", always_xy=True)
        except Exception as exc:
            log.warning("No se pudo preparar transformación WGS84: %s", exc)

    # Construir índice GDF por id para búsqueda O(1)
    gdf_by_id: dict = {}
    if gdf is not None:
        id_col = "id" if "id" in gdf.columns else None
        for idx, row in gdf.iterrows():
            obj_id = row["id"] if id_col else idx
            gdf_by_id[obj_id] = row

    # Determinar campos
    fields = _FIELDS_BASE + (_FIELDS_GEO if gdf is not None else [])

    rows = []
    for area in areas:
        obj_id = area["id"]
        row = {
            "id": obj_id,
            "area_m2": round(area["area_m2"], 2),
            "area_px": area["area_px"],
            "centroid_row": area["centroid_px"][0],
            "centroid_col": area["centroid_px"][1],
        }

        if gdf is not None and obj_id in gdf_by_id:
            geom_row = gdf_by_id[obj_id]
            centroid = geom_row.geometry.centroid
            row["centroid_x_m"] = round(centroid.x, 2)
            row["centroid_y_m"] = round(centroid.y, 2)
            if crs_transformer:
                try:
                    lon, lat = crs_transformer.transform(centroid.x, centroid.y)
                    row["centroid_lon"] = round(lon, 7)
                    row["centroid_lat"] = round(lat, 7)
                except Exception:
                    row["centroid_lon"] = ""
                    row["centroid_lat"] = ""
            else:
                row["centroid_lon"] = round(centroid.x, 7)
                row["centroid_lat"] = round(centroid.y, 7)
        elif gdf is not None:
            row.update({"centroid_x_m": "", "centroid_y_m": "",
                        "centroid_lon": "", "centroid_lat": ""})

        rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    log.info(
        "CSV exportado: %s | %d objeto(s) | área total: %.1f m²",
        output_path,
        len(rows),
        sum(r["area_m2"] for r in rows),
    )
    return output_path
