"""Preparación de tiles de entrenamiento para el modelo U-Net.

Genera pares (imagen, máscara) en formato .npy a partir de:
  - Imágenes Sentinel-2 en formato GeoTIFF multibanda (6 bandas apiladas:
    B02, B03, B04, B08, B11, B12), como las que produce safe_loader.py.
  - Polígonos de edificios en formato GeoPackage, GeoJSON o Shapefile.

El resultado se guarda en::

    output_dir/
        images/   ← float32 (6, TILE_SIZE, TILE_SIZE) normalizado [0, 1]
        masks/    ← uint8   (TILE_SIZE, TILE_SIZE), 1=techo, 0=fondo

Estos archivos son directamente consumibles por RoofDataset en trainer.py.

────────────────────────────────────────────────────
 ¿Dónde descargar los polígonos de edificios?
────────────────────────────────────────────────────
Opción A — Google Open Buildings V3 (recomendada, CC BY-4.0):
  https://sites.research.google/gr/open-buildings/
  Filtrá por país Argentina y descargá el GeoPackage.
  Columna de confianza: "confidence"

Opción B — VIDA combined dataset (Google+MS+OSM, ODbL):
  https://source.coop/vida/google-microsoft-osm-open-buildings
  Particionado por país, archivo para Argentina: ARG.gpkg

Opción C — Microsoft ML Footprints (ODbL):
  https://github.com/microsoft/GlobalMLBuildingFootprints
  No tiene columna de confianza; se usan todos los polígonos.

Opción D — OpenStreetMap (Geofabrik Argentina):
  https://download.geofabrik.de/south-america/argentina.html
  Extraé la capa "buildings" del archivo .osm.pbf con ogr2ogr.

────────────────────────────────────────────────────
 Uso
────────────────────────────────────────────────────
  python scripts/prepare_tiles.py \\
      --buildings data/pretrain/ARG.gpkg \\
      --tiles-dir data/cache/ \\
      --output-dir data/pretrain/ \\
      [--confidence 0.7] \\
      [--tile-size 256] \\
      [--stride 128] \\
      [--min-roof-pct 1.0] \\
      [--max-nodata-pct 20.0] \\
      [--glob "*_stacked.tif"]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_tiles")

# ---------------------------------------------------------------------------
# Constantes por defecto
# ---------------------------------------------------------------------------
DEFAULT_TILE_SIZE = 256
DEFAULT_STRIDE = 128          # 50 % de solapamiento entre tiles
DEFAULT_MIN_ROOF_PCT = 1.0    # % mínimo de píxeles de techo para conservar un tile
DEFAULT_MAX_NODATA_PCT = 20.0 # % máximo de NaN/nodata para conservar un tile
DEFAULT_CONFIDENCE = 0.70     # filtro mínimo de confianza (Open Buildings)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--buildings", required=True, type=Path,
        help="Archivo de polígonos de edificios (.gpkg, .geojson, .shp)",
    )
    p.add_argument(
        "--tiles-dir", required=True, type=Path,
        help="Directorio con GeoTIFFs Sentinel-2 multibanda",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("data/pretrain"),
        help="Directorio de salida para los tiles .npy (default: data/pretrain)",
    )
    p.add_argument(
        "--confidence", type=float, default=DEFAULT_CONFIDENCE,
        help=f"Confianza mínima para Open Buildings (default: {DEFAULT_CONFIDENCE})",
    )
    p.add_argument(
        "--tile-size", type=int, default=DEFAULT_TILE_SIZE,
        help=f"Tamaño del tile en píxeles (default: {DEFAULT_TILE_SIZE})",
    )
    p.add_argument(
        "--stride", type=int, default=DEFAULT_STRIDE,
        help=f"Paso entre tiles en píxeles (default: {DEFAULT_STRIDE})",
    )
    p.add_argument(
        "--min-roof-pct", type=float, default=DEFAULT_MIN_ROOF_PCT,
        help=f"Porcentaje mínimo de píxeles de techo en un tile (default: {DEFAULT_MIN_ROOF_PCT})",
    )
    p.add_argument(
        "--max-nodata-pct", type=float, default=DEFAULT_MAX_NODATA_PCT,
        help=f"Porcentaje máximo de nodata en un tile (default: {DEFAULT_MAX_NODATA_PCT})",
    )
    p.add_argument(
        "--glob", type=str, default="*_stacked.tif",
        help='Patrón glob para buscar GeoTIFFs en --tiles-dir (default: "*_stacked.tif")',
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Carga y filtrado de edificios
# ---------------------------------------------------------------------------

def _load_buildings(buildings_path: Path, confidence_min: float):
    """Carga los polígonos de edificios y aplica el filtro de confianza.

    Returns:
        GeoDataFrame en CRS WGS84 (EPSG:4326).
    """
    try:
        import geopandas as gpd
    except ImportError:
        log.error("geopandas no está instalado. Ejecutá: pip install geopandas")
        sys.exit(1)

    log.info("Cargando polígonos de edificios: %s", buildings_path)
    gdf = gpd.read_file(buildings_path)
    log.info("  Polígonos cargados: %d", len(gdf))

    # Filtrar por confianza si la columna existe
    conf_cols = [c for c in gdf.columns if "confidence" in c.lower() or "conf" == c.lower()]
    if conf_cols:
        col = conf_cols[0]
        before = len(gdf)
        gdf = gdf[gdf[col] >= confidence_min]
        log.info(
            "  Filtro confianza '%s' ≥ %.2f: %d → %d polígonos",
            col, confidence_min, before, len(gdf),
        )
    else:
        log.warning(
            "  No se encontró columna de confianza en el archivo. "
            "Se usan todos los polígonos."
        )

    # Asegurar CRS WGS84
    if gdf.crs is None:
        log.warning("  GeoDataFrame sin CRS definido. Asignando WGS84.")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    log.info("  Polígonos finales tras filtros: %d", len(gdf))
    return gdf


# ---------------------------------------------------------------------------
# Rasterización
# ---------------------------------------------------------------------------

def _rasterize_buildings(gdf_utm, transform, height: int, width: int) -> np.ndarray:
    """Rasteriza polígonos de edificios al grid del GeoTIFF.

    Args:
        gdf_utm: GeoDataFrame en el CRS del raster (UTM).
        transform: Affine transform del raster.
        height, width: Dimensiones del raster de salida.

    Returns:
        Array uint8 ``(height, width)``, 1 = edificio, 0 = fondo.
    """
    from rasterio.features import rasterize as rio_rasterize

    if len(gdf_utm) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    shapes = [
        (geom, 1)
        for geom in gdf_utm.geometry
        if geom is not None and not geom.is_empty
    ]
    if not shapes:
        return np.zeros((height, width), dtype=np.uint8)

    return rio_rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )


# ---------------------------------------------------------------------------
# Extracción de tiles
# ---------------------------------------------------------------------------

def _extract_tiles(
    image: np.ndarray,
    mask: np.ndarray,
    tile_size: int,
    stride: int,
    min_roof_pct: float,
    max_nodata_pct: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extrae tiles válidos de la imagen y máscara con ventana deslizante.

    Args:
        image: Array float32 ``(6, H, W)`` normalizado [0, 1], NaN en nodata.
        mask: Array uint8 ``(H, W)``, 1=edificio, 0=fondo.
        tile_size: Tamaño del tile en píxeles.
        stride: Paso entre tiles en píxeles.
        min_roof_pct: Porcentaje mínimo de píxeles de techo para conservar.
        max_nodata_pct: Porcentaje máximo de NaN para conservar.

    Returns:
        Lista de tuplas ``(img_tile, mask_tile)``.
    """
    _, H, W = image.shape
    tiles = []
    total = skip_nodata = skip_no_roof = 0

    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            img_tile = image[:, y:y + tile_size, x:x + tile_size]
            msk_tile = mask[y:y + tile_size, x:x + tile_size]
            total += 1

            # Descartar tiles con exceso de nodata
            nan_pct = 100.0 * np.isnan(img_tile).mean()
            if nan_pct > max_nodata_pct:
                skip_nodata += 1
                continue

            # Descartar tiles sin suficientes techos
            roof_pct = 100.0 * msk_tile.mean()
            if roof_pct < min_roof_pct:
                skip_no_roof += 1
                continue

            tiles.append((img_tile.copy(), msk_tile.copy()))

    log.debug(
        "  Tiles: total=%d | descartados nodata=%d | sin_techo=%d | válidos=%d",
        total, skip_nodata, skip_no_roof, len(tiles),
    )
    return tiles


# ---------------------------------------------------------------------------
# Procesamiento por GeoTIFF
# ---------------------------------------------------------------------------

def _process_geotiff(
    tif_path: Path,
    gdf_wgs84,
    args: argparse.Namespace,
    output_dir: Path,
    tile_counter: list,
) -> int:
    """Carga un GeoTIFF, rasteriza edificios y guarda los tiles.

    Returns:
        Número de tiles generados para este GeoTIFF.
    """
    try:
        import rasterio
    except ImportError:
        log.error("rasterio no está instalado.")
        sys.exit(1)

    log.info("Procesando: %s", tif_path.name)

    with rasterio.open(tif_path) as src:
        array = src.read().astype(np.float32)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds

    # Normalizar DN → reflectancia [0, 1]
    array = array / 10_000.0
    array = np.clip(array, 0.0, 1.0)

    # Marcar como NaN los píxeles completamente nulos (nodata S2)
    nodata_px = (array == 0.0).all(axis=0)
    array[:, nodata_px] = np.nan

    _, H, W = array.shape
    log.info("  Dimensiones: %d × %d px | Bandas: %d", W, H, array.shape[0])

    # Recortar edificios al bbox del GeoTIFF (en WGS84)
    try:
        from pyproj import Transformer
        if "4326" not in str(crs):
            t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon_min, lat_min = t.transform(bounds.left, bounds.bottom)
            lon_max, lat_max = t.transform(bounds.right, bounds.top)
        else:
            lon_min, lat_min = bounds.left, bounds.bottom
            lon_max, lat_max = bounds.right, bounds.top
    except Exception as exc:
        log.warning("  No se pudo calcular bbox WGS84: %s. Usando todos los edificios.", exc)
        lon_min, lat_min, lon_max, lat_max = -180, -90, 180, 90

    gdf_clip = gdf_wgs84.cx[lon_min:lon_max, lat_min:lat_max]
    if len(gdf_clip) == 0:
        log.warning("  Sin edificios en el bbox de este GeoTIFF. Saltando.")
        return 0

    # Reproyectar edificios al CRS del raster
    gdf_utm = gdf_clip.to_crs(crs)

    # Rasterizar edificios
    mask = _rasterize_buildings(gdf_utm, transform, H, W)
    log.info(
        "  Edificios en bbox: %d | Px techo: %d (%.1f%%)",
        len(gdf_clip), mask.sum(), 100.0 * mask.mean(),
    )

    # Extraer tiles válidos
    tiles = _extract_tiles(
        array, mask,
        args.tile_size, args.stride,
        args.min_roof_pct, args.max_nodata_pct,
    )

    # Guardar tiles
    img_dir = output_dir / "images"
    msk_dir = output_dir / "masks"
    for img_tile, msk_tile in tiles:
        idx = tile_counter[0]
        name = f"{idx:05d}.npy"
        np.save(img_dir / name, img_tile)
        np.save(msk_dir / name, msk_tile)
        tile_counter[0] += 1

    log.info("  → %d tile(s) guardados.", len(tiles))
    return len(tiles)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Validar inputs
    if not args.buildings.exists():
        log.error("Archivo de edificios no encontrado: %s", args.buildings)
        sys.exit(1)

    tif_files = sorted(args.tiles_dir.glob(args.glob))
    if not tif_files:
        log.error(
            "No se encontraron GeoTIFFs con patrón '%s' en: %s",
            args.glob, args.tiles_dir,
        )
        sys.exit(1)

    log.info("GeoTIFFs encontrados: %d", len(tif_files))
    for f in tif_files:
        log.info("  %s", f.name)

    # Crear directorios de salida
    (args.output_dir / "images").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "masks").mkdir(parents=True, exist_ok=True)

    # Cargar edificios una sola vez (puede ser grande, se filtra por bbox en cada GeoTIFF)
    gdf = _load_buildings(args.buildings, args.confidence)

    # Procesar cada GeoTIFF
    tile_counter = [0]
    for tif_path in tif_files:
        _process_geotiff(tif_path, gdf, args, args.output_dir, tile_counter)

    total = tile_counter[0]
    log.info("=" * 60)
    log.info("Preparación completa | Total tiles generados: %d", total)
    log.info("Directorio de salida: %s", args.output_dir.resolve())

    if total == 0:
        log.error(
            "No se generó ningún tile. Posibles causas:\n"
            "  - Los polígonos de edificios no cubren el área de los GeoTIFFs.\n"
            "  - El filtro de confianza (%.2f) es demasiado estricto.\n"
            "  - El filtro de techo mínimo (%.1f%%) elimina todos los tiles.",
            args.confidence, args.min_roof_pct,
        )
    elif total < 20:
        log.warning(
            "Pocos tiles generados (%d). Para un preentrenamiento robusto se "
            "recomiendan ≥ 50. Considerá:\n"
            "  - Agregar más GeoTIFFs Sentinel-2 de la zona.\n"
            "  - Reducir --tile-size o --min-roof-pct.\n"
            "  - Ampliar el área geográfica de descarga.",
            total,
        )
    else:
        log.info(
            "Siguiente paso:\n"
            "  python scripts/pretrain_unet.py --tiles-dir %s",
            args.output_dir,
        )


if __name__ == "__main__":
    main()
