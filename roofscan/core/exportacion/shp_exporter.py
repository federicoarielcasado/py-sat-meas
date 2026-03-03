"""Exportación de detecciones a Shapefile (ESRI).

El formato Shapefile limita los nombres de columna a 10 caracteres.
Este módulo aplica un renombrado preventivo para los campos canónicos
de RoofScan que podrían superar ese límite.

Uso típico::

    from roofscan.core.exportacion.shp_exporter import export_shapefile

    path = export_shapefile(gdf, "data/output/techos.shp")
    # Genera también: techos.dbf, techos.shx, techos.prj
"""

from pathlib import Path

import geopandas as gpd

# Renombrado de columnas canónicas para respetar el límite de 10 chars del formato.
# Solo se aplica a las columnas que efectivamente estén presentes.
_SHP_RENAME: dict[str, str] = {
    "centroid_x_m": "cent_x_m",
    "centroid_y_m": "cent_y_m",
    "centroid_lon": "cent_lon",
    "centroid_lat": "cent_lat",
}


def export_shapefile(
    gdf: gpd.GeoDataFrame,
    output_path: str | Path,
) -> Path:
    """Guarda un GeoDataFrame en formato Shapefile de ESRI.

    Aplica renombrado de columnas para respetar el límite de 10 caracteres
    del formato. Si el directorio destino no existe, se crea automáticamente.

    Args:
        gdf: GeoDataFrame con las geometrías y atributos a exportar.
        output_path: Ruta completa del archivo de salida, incluyendo el
            nombre de archivo y la extensión ``.shp``. Los archivos
            auxiliares (``.dbf``, ``.shx``, ``.prj``) se generan en
            la misma carpeta.

    Returns:
        Path al archivo ``.shp`` guardado.

    Raises:
        ValueError: Si el GeoDataFrame está vacío.
        OSError: Si no se puede escribir en la ruta indicada.
    """
    if gdf.empty:
        raise ValueError(
            "El GeoDataFrame está vacío; no hay geometrías para exportar."
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rename = {k: v for k, v in _SHP_RENAME.items() if k in gdf.columns}
    export_gdf = gdf.rename(columns=rename) if rename else gdf

    export_gdf.to_file(str(path), driver="ESRI Shapefile", encoding="utf-8")
    return path
