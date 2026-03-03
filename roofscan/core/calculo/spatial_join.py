"""Intersección espacial entre polígonos de techos detectados y parcelas catastrales.

Toma el resultado de la detección (GeoDataFrame de techos) y el catastro
(GeoDataFrame de parcelas) y produce una tabla resumen con el área de techo
por parcela, lista para exportar como CSV.

Uso típico::

    from roofscan.core.calculo.spatial_join import join_roofs_to_parcelas

    resultado = join_roofs_to_parcelas(gdf_techos, gdf_parcelas)
    resultado.to_csv("mensura_lujan.csv", index=False)

Columnas de salida::

    nomenclatura      # código catastral de la parcela
    partido           # nombre del partido (si disponible)
    seccion           # sección (si disponible)
    manzana           # manzana (si disponible)
    parcela_num       # número de parcela (si disponible)
    area_parcela_m2   # área total de la parcela en m²
    area_techos_m2    # área de techo detectada dentro de la parcela en m²
    n_techos          # número de objetos de techo detectados
    pct_cubierto      # porcentaje del área de parcela cubierto por techos
    geometry          # geometría de la parcela (WGS84)
"""

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# CRS de trabajo: UTM zona 21S (Luján ~59°W → zona 21S 60°W-54°W)
_WORK_CRS = "EPSG:32721"


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def join_roofs_to_parcelas(
    gdf_roofs: "GeoDataFrame",
    gdf_parcelas: "GeoDataFrame",
    work_crs: str = _WORK_CRS,
    min_intersection_m2: float = 1.0,
) -> "GeoDataFrame":
    """Calcula el área de techo dentro de cada parcela catastral.

    Realiza una intersección espacial entre los polígonos de techos detectados
    y los polígonos catastrales. Para cada parcela, suma las áreas de los
    techos que caen dentro de ella (parcial o totalmente).

    Args:
        gdf_roofs: GeoDataFrame con los polígonos de techos. Debe tener
                   columnas ``geometry`` y ``area_m2``.
                   Generado por :func:`roofscan.core.calculo.geometry_merger.labels_to_geodataframe`.
        gdf_parcelas: GeoDataFrame con los polígonos catastrales.
                      Generado por :func:`roofscan.core.ingesta.wfs_arba.get_parcelas`.
        work_crs: CRS proyectado para cálculos de área (debe estar en metros).
                  Por defecto EPSG:32720 (UTM 20S, Argentina).
        min_intersection_m2: Intersección mínima en m² para considerar que un
                             techo pertenece a una parcela. Filtra solapamientos
                             marginales por bordes. Default: 1.0 m².

    Returns:
        GeoDataFrame con una fila por parcela y las columnas descritas arriba,
        en CRS WGS84 (EPSG:4326).

    Raises:
        ImportError: Si geopandas o shapely no están instalados.
        ValueError: Si alguno de los GeoDataFrames está vacío.
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("geopandas no está instalado.") from exc

    if len(gdf_roofs) == 0:
        raise ValueError("gdf_roofs está vacío: no se detectaron techos en la imagen.")
    if len(gdf_parcelas) == 0:
        raise ValueError("gdf_parcelas está vacío: no hay parcelas para procesar.")

    log.info(
        "Intersección espacial | techos=%d | parcelas=%d",
        len(gdf_roofs), len(gdf_parcelas),
    )

    # Reprojectar ambos al CRS de trabajo (metros) para cálculos de área
    roofs_utm = _ensure_crs(gdf_roofs, work_crs)
    parcelas_utm = _ensure_crs(gdf_parcelas, work_crs)

    # Calcular área de parcelas en m² (desde geometría proyectada)
    parcelas_utm = parcelas_utm.copy().reset_index(drop=True)
    parcelas_utm["area_parcela_m2"] = parcelas_utm.geometry.area

    # Intersección espacial: left=parcelas, right=techos
    # → geometry del resultado = geometría de la parcela (left)
    # → index_right = índice del techo candidato en roofs_utm
    log.info("  Ejecutando sjoin (intersects)…")
    joined = gpd.sjoin(
        parcelas_utm,
        roofs_utm[["geometry", "area_m2", "roof_id"]].reset_index(drop=True),
        how="left",
        predicate="intersects",
    )
    # roofs_utm con índice 0-based para lookup por posición
    roofs_indexed = roofs_utm.reset_index(drop=True)

    # Para cada par (techo, parcela), calcular la intersección real
    log.info("  Calculando intersecciones exactas…")
    result_rows = []

    for idx_parcela, parcela_row in parcelas_utm.iterrows():
        parcela_geom = parcela_row.geometry
        # Con how="left", el índice de joined ES el índice de parcelas_utm
        rows = joined.loc[joined.index == idx_parcela]
        techo_indices = rows["index_right"].dropna().astype(int)

        techo_area_total = 0.0
        n_techos = 0

        for roof_idx in techo_indices:
            try:
                roof_geom = roofs_indexed.loc[roof_idx, "geometry"]
                inter = parcela_geom.intersection(roof_geom)
                inter_area = inter.area
            except Exception:
                inter_area = 0.0

            if inter_area >= min_intersection_m2:
                techo_area_total += inter_area
                n_techos += 1

        area_parcela = parcela_row.get("area_parcela_m2", parcela_geom.area)
        pct = (techo_area_total / area_parcela * 100) if area_parcela > 0 else 0.0

        row: dict[str, Any] = {
            # Atributos catastrales
            "nomenclatura":    parcela_row.get("nomenclatura", ""),
            "partido":         parcela_row.get("partido", ""),
            "seccion":         parcela_row.get("seccion", ""),
            "manzana":         parcela_row.get("manzana", ""),
            "parcela_num":     parcela_row.get("parcela", ""),
            # Áreas
            "area_parcela_m2": round(area_parcela, 2),
            "area_techos_m2":  round(techo_area_total, 2),
            "n_techos":        n_techos,
            "pct_cubierto":    round(pct, 2),
            # Geometría
            "geometry":        parcela_geom,
        }
        result_rows.append(row)

    if not result_rows:
        import geopandas as gpd
        return gpd.GeoDataFrame(columns=[
            "nomenclatura", "partido", "seccion", "manzana", "parcela_num",
            "area_parcela_m2", "area_techos_m2", "n_techos", "pct_cubierto", "geometry",
        ], crs=work_crs)

    import geopandas as gpd
    result = gpd.GeoDataFrame(result_rows, crs=work_crs)
    result = result.to_crs("EPSG:4326")

    n_con_techo = (result["n_techos"] > 0).sum()
    log.info(
        "  Resultado: %d parcelas | %d con techo detectado | área media: %.1f m²",
        len(result), n_con_techo,
        result.loc[result["n_techos"] > 0, "area_techos_m2"].mean()
        if n_con_techo > 0 else 0.0,
    )

    return result.reset_index(drop=True)


def export_mensura_csv(
    result: "GeoDataFrame",
    output_path: "str | Path",
    include_geometry: bool = False,
) -> "Path":
    """Exporta el resultado de la mensura a CSV.

    Args:
        result: GeoDataFrame retornado por :func:`join_roofs_to_parcelas`.
        output_path: Ruta del archivo CSV de salida.
        include_geometry: Si ``True``, incluye las coordenadas del centroide
                          de cada parcela (lat, lon) en el CSV.

    Returns:
        :class:`pathlib.Path` al CSV generado.
    """
    from pathlib import Path
    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = result.drop(columns=["geometry"], errors="ignore").copy()

    if include_geometry and "geometry" in result.columns:
        centroids = result.geometry.to_crs("EPSG:4326").centroid
        df["centroide_lat"] = centroids.y.round(6)
        df["centroide_lon"] = centroids.x.round(6)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    log.info("CSV exportado: %s (%d filas)", output_path, len(df))
    return output_path


def summary_stats(result: "GeoDataFrame") -> dict[str, Any]:
    """Calcula estadísticas resumidas del resultado de la mensura.

    Args:
        result: GeoDataFrame retornado por :func:`join_roofs_to_parcelas`.

    Returns:
        Dict con métricas clave::

            {
                "total_parcelas": int,
                "parcelas_con_techo": int,
                "parcelas_sin_techo": int,
                "area_total_techos_m2": float,
                "area_media_techo_m2": float,
                "pct_cubierto_medio": float,
                "pct_cubierto_max": float,
            }
    """
    con_techo = result[result["n_techos"] > 0]
    return {
        "total_parcelas":        len(result),
        "parcelas_con_techo":    len(con_techo),
        "parcelas_sin_techo":    len(result) - len(con_techo),
        "area_total_techos_m2":  round(result["area_techos_m2"].sum(), 2),
        "area_media_techo_m2":   round(con_techo["area_techos_m2"].mean(), 2) if len(con_techo) else 0.0,
        "pct_cubierto_medio":    round(con_techo["pct_cubierto"].mean(), 2) if len(con_techo) else 0.0,
        "pct_cubierto_max":      round(result["pct_cubierto"].max(), 2),
    }


# ---------------------------------------------------------------------------
# Helper interno
# ---------------------------------------------------------------------------

def _ensure_crs(gdf: "GeoDataFrame", target_crs: str) -> "GeoDataFrame":
    """Reproyecta un GeoDataFrame al CRS objetivo si es necesario."""
    if gdf.crs is None:
        log.warning("GeoDataFrame sin CRS. Asignando EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf
