"""Descarga las parcelas catastrales de Luján (o cualquier partido) desde el WFS de ARBA.

El WFS ``https://geo.arba.gov.ar/geoserver/idera/wfs`` está activo y retorna
polígonos en formato GeoJSON. Este script descarga todas las parcelas de un
bbox en páginas y las guarda en ``data/catastro/lujan_parcelas.gpkg``.

Uso:
    # Luján (bbox por defecto):
    python scripts/download_catastro.py

    # Bbox personalizado:
    python scripts/download_catastro.py --bbox "-59.15,-34.70,-58.90,-34.45"

    # Solo contar sin descargar:
    python scripts/download_catastro.py --count-only

Salida:
    data/catastro/lujan_parcelas.gpkg  (~20-50 MB, ~140k parcelas)

Tiempo estimado: 3-8 minutos según la velocidad del servidor.

Notas técnicas:
- El WFS devuelve geometrías en EPSG:5347 (POSGAR 2007 / Argentina 7).
  El script las reprojecta automáticamente a WGS84 (EPSG:4326) al guardar.
- Columnas: cca (código catastral), tpa (tipo: Urbano/Rural), ara1 (área m²),
  pda (partida), sag (fuente).
"""

import argparse
import io
import logging
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_catastro")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_WFS_URL = "https://geo.arba.gov.ar/geoserver/idera/wfs"
_TYPENAME = "idera:Parcela"
_PAGE_SIZE = 5_000        # features por página (el servidor aguanta hasta ~10k)
_TIMEOUT = 90             # segundos por request
_RETRY_MAX = 3            # reintentos ante error de red

# Bbox del partido de Luján, Buenos Aires
_LUJAN_BBOX = (-59.15, -34.70, -58.90, -34.45)

_OUTPUT_DIR = Path("data/catastro")
_OUTPUT_NAME = "lujan_parcelas"


# ---------------------------------------------------------------------------
# Helpers WFS
# ---------------------------------------------------------------------------

def _wfs_count(bbox: tuple[float, float, float, float]) -> int:
    """Retorna el número total de features en el bbox dado."""
    lon_min, lat_min, lon_max, lat_max = bbox
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": _TYPENAME,
        "outputFormat": "application/json",
        "resultType": "hits",
        "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max},EPSG:4326",
    }
    resp = requests.get(_WFS_URL, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    # El servidor retorna XML con el atributo numberMatched
    import xml.etree.ElementTree as ET
    root = ET.fromstring(resp.content)
    return int(root.get("numberMatched", 0))


def _wfs_page(
    bbox: tuple[float, float, float, float],
    start_index: int,
    count: int,
) -> bytes:
    """Descarga una página de features desde el WFS."""
    lon_min, lat_min, lon_max, lat_max = bbox
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": _TYPENAME,
        "outputFormat": "application/json",
        "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max},EPSG:4326",
        "count": count,
        "startIndex": start_index,
    }
    for attempt in range(1, _RETRY_MAX + 1):
        try:
            resp = requests.get(_WFS_URL, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            # Verificar que la respuesta sea JSON válido
            if resp.content.startswith(b"{"):
                return resp.content
            # Si retorna XML de excepción, lanzar error descriptivo
            raise ConnectionError(f"El servidor devolvió respuesta inesperada: {resp.content[:200]}")
        except Exception as exc:
            if attempt < _RETRY_MAX:
                wait = 5 * attempt
                log.warning("  Intento %d/%d fallido: %s. Reintentando en %ds…",
                             attempt, _RETRY_MAX, exc, wait)
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Descarga completa con paginación
# ---------------------------------------------------------------------------

def download_parcelas(
    bbox: tuple[float, float, float, float],
    output_dir: Path,
    output_name: str,
) -> Path:
    """Descarga todas las parcelas del bbox con paginación y guarda como GeoPackage.

    Returns:
        Path al archivo .gpkg generado.
    """
    try:
        import geopandas as gpd
        import pandas as pd
    except ImportError:
        log.error("geopandas no está instalado. Instalalo con: pip install geopandas")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    gpkg_path = output_dir / f"{output_name}.gpkg"

    # --- Contar total ---
    log.info("Consultando total de parcelas en el bbox…")
    try:
        total = _wfs_count(bbox)
    except Exception as exc:
        log.error("No se pudo consultar el WFS: %s", exc)
        log.error("Verificá la conexión a internet y que el servidor esté disponible.")
        sys.exit(1)

    if total == 0:
        log.error("El WFS no retornó ninguna parcela para el bbox indicado.")
        sys.exit(1)

    log.info("  Total de parcelas a descargar: %d", total)
    pages = (total + _PAGE_SIZE - 1) // _PAGE_SIZE
    log.info("  Páginas necesarias: %d (de %d features c/u)", pages, _PAGE_SIZE)

    # --- Descargar en páginas ---
    all_gdfs = []
    t0 = time.time()

    for page_n in range(pages):
        start = page_n * _PAGE_SIZE
        log.info("  Página %d/%d | start=%d…", page_n + 1, pages, start)

        try:
            raw = _wfs_page(bbox, start, _PAGE_SIZE)
        except Exception as exc:
            log.error("  Error irrecuperable en página %d: %s", page_n + 1, exc)
            if all_gdfs:
                log.warning("  Guardando las %d páginas descargadas hasta ahora.",
                             len(all_gdfs))
                break
            sys.exit(1)

        gdf_page = gpd.read_file(io.BytesIO(raw))
        if len(gdf_page) == 0:
            log.info("  Página vacía — fin de datos.")
            break

        all_gdfs.append(gdf_page)
        downloaded = sum(len(g) for g in all_gdfs)
        elapsed = time.time() - t0
        rate = downloaded / elapsed if elapsed > 0 else 0
        eta = (total - downloaded) / rate if rate > 0 else float("inf")
        log.info("  Descargadas: %d/%d (%.0f feat/s | ETA: %.0fs)",
                 downloaded, total, rate, eta)

    if not all_gdfs:
        log.error("No se descargó ninguna página.")
        sys.exit(1)

    # --- Combinar y guardar ---
    log.info("Combinando %d páginas (%d features total)…",
             len(all_gdfs), sum(len(g) for g in all_gdfs))

    gdf = pd.concat(all_gdfs, ignore_index=True)
    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry")

    # Los datos de ARBA vienen en EPSG:5347 (POSGAR 2007 / Argentina 7).
    # Reprojectar a WGS84 para compatibilidad con el resto del sistema.
    if gdf.crs is None:
        log.warning("CRS no detectado — asignando EPSG:5347 (POSGAR 2007 Argentina 7).")
        gdf = gdf.set_crs("EPSG:5347")
    if gdf.crs.to_epsg() != 4326:
        log.info("Reprojectando %s → EPSG:4326 (WGS84)…", gdf.crs)
        gdf = gdf.to_crs("EPSG:4326")

    log.info("Guardando GeoPackage: %s", gpkg_path)
    gdf.to_file(str(gpkg_path), driver="GPKG")

    size_mb = gpkg_path.stat().st_size / 1_048_576
    elapsed_total = time.time() - t0
    log.info("  Listo: %.1f MB en %.0f segundos.", size_mb, elapsed_total)

    return gpkg_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--bbox", type=str,
        default=",".join(str(v) for v in _LUJAN_BBOX),
        help=(
            "Bbox WGS84 'lon_min,lat_min,lon_max,lat_max'. "
            f"Default: {','.join(str(v) for v in _LUJAN_BBOX)} (partido de Luján)."
        ),
    )
    p.add_argument(
        "--output-dir", type=Path, default=_OUTPUT_DIR,
        help=f"Directorio de salida (default: {_OUTPUT_DIR}).",
    )
    p.add_argument(
        "--output-name", type=str, default=_OUTPUT_NAME,
        help=f"Nombre del archivo de salida sin extensión (default: {_OUTPUT_NAME}).",
    )
    p.add_argument(
        "--count-only", action="store_true",
        help="Solo contar parcelas disponibles, sin descargar.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Parsear bbox
    try:
        parts = [float(x.strip()) for x in args.bbox.split(",")]
        if len(parts) != 4:
            raise ValueError("Debe tener 4 valores")
        bbox = tuple(parts)
    except ValueError as exc:
        log.error("Bbox inválido: %s", exc)
        sys.exit(1)

    log.info("WFS: %s", _WFS_URL)
    log.info("Bbox: lon_min=%.4f lat_min=%.4f lon_max=%.4f lat_max=%.4f", *bbox)

    # Solo contar
    if args.count_only:
        log.info("Consultando total de parcelas…")
        try:
            total = _wfs_count(bbox)
            log.info("Total de parcelas en el bbox: %d", total)
        except Exception as exc:
            log.error("Error: %s", exc)
            sys.exit(1)
        return

    # Descargar
    gpkg_path = download_parcelas(bbox, args.output_dir, args.output_name)

    log.info("")
    log.info("=" * 60)
    log.info("Catastro descargado. Para correr la mensura masiva:")
    log.info("")
    log.info("  python scripts/batch_mensura.py \\")
    log.info("      --image data/cache/S2A_..._stacked.tif \\")
    log.info("      --parcelas %s \\", gpkg_path.resolve())
    log.info("      --output data/output/mensura_lujan.csv \\")
    log.info("      --output-geojson --classify")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
