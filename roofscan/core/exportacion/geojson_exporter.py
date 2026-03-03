"""Exportación de detecciones a GeoJSON.

Wrapper sobre geopandas que garantiza consistencia de formato y manejo
explícito de errores, alineado con el resto de módulos de exportación.

Uso típico::

    from roofscan.core.exportacion.geojson_exporter import export_geojson

    path = export_geojson(gdf, "data/output/techos.geojson")
"""

from pathlib import Path

import geopandas as gpd


def export_geojson(
    gdf: gpd.GeoDataFrame,
    output_path: str | Path,
) -> Path:
    """Guarda un GeoDataFrame en formato GeoJSON.

    El archivo se crea en el directorio indicado por ``output_path``.
    Si el directorio no existe, se crea automáticamente.

    Args:
        gdf: GeoDataFrame con las geometrías y atributos a exportar.
            Debe contener al menos una columna ``geometry``.
        output_path: Ruta completa del archivo de salida, incluyendo el
            nombre de archivo y la extensión ``.geojson``.

    Returns:
        Path al archivo GeoJSON guardado.

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
    gdf.to_file(str(path), driver="GeoJSON")
    return path
