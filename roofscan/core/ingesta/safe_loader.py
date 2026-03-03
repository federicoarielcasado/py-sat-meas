"""Conversión de escenas Sentinel-2 .SAFE a GeoTIFF multibanda.

El downloader descarga las escenas en formato .SAFE (estructura de carpetas
con archivos JP2 individuales por banda). Este módulo apila las bandas
requeridas en un único GeoTIFF, remuestreando las bandas de 20 m a 10 m.

Uso típico::

    from roofscan.core.ingesta.safe_loader import safe_to_geotiff
    from roofscan.config import CACHE_DIR

    geotiff_path = safe_to_geotiff(Path("escena.SAFE"), CACHE_DIR)
    # geotiff_path → data/cache/S2A_..._stacked.tif  (B02,B03,B04,B08,B11,B12)
    # También genera: data/cache/S2A_..._SCL.tif
"""

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Resolución nativa de cada banda Sentinel-2 L2A
_BAND_RESOLUTION: dict[str, str] = {
    "B02": "10m", "B03": "10m", "B04": "10m", "B08": "10m",
    "B11": "20m", "B12": "20m", "SCL": "20m",
}

# Bandas espectrales a apilar en el GeoTIFF principal (en este orden → canales 1..6)
_DEFAULT_SPECTRAL = ["B02", "B03", "B04", "B08", "B11", "B12"]


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def find_band_files(safe_path: Path, bands: list[str] | None = None) -> dict[str, Path]:
    """Localiza los archivos JP2/TIF de cada banda dentro de la estructura .SAFE.

    Args:
        safe_path: Ruta a la carpeta ``.SAFE`` de la escena.
        bands: Lista de nombres de banda (ej. ``["B02", "SCL"]``).
               Por defecto busca todas las bandas en ``_BAND_RESOLUTION``.

    Returns:
        Dict ``{banda: Path}`` con los archivos encontrados. Las bandas
        no encontradas se omiten del resultado.

    Raises:
        FileNotFoundError: Si la carpeta GRANULE no existe dentro de ``.SAFE``.
    """
    if bands is None:
        bands = list(_BAND_RESOLUTION.keys())

    granule_dir = Path(safe_path) / "GRANULE"
    if not granule_dir.exists():
        raise FileNotFoundError(
            f"No se encontró la carpeta GRANULE en: {safe_path}\n"
            "Verificá que la descarga esté completa."
        )

    found: dict[str, Path] = {}

    for granule in granule_dir.iterdir():
        if not granule.is_dir():
            continue
        img_data = granule / "IMG_DATA"
        if not img_data.exists():
            continue

        for band in bands:
            if band in found:
                continue  # ya encontrada en otro granule (no debería ocurrir)
            res = _BAND_RESOLUTION.get(band, "10m")
            res_folder = img_data / f"R{res}"
            if not res_folder.exists():
                continue

            # Patrones de búsqueda: primero específico, luego genérico
            patterns = [f"*_{band}_{res}.jp2", f"*_{band}_{res}.tif", f"*_{band}.jp2"]
            for pat in patterns:
                matches = list(res_folder.glob(pat))
                if matches:
                    found[band] = matches[0]
                    break

    missing = set(bands) - set(found.keys())
    if missing:
        log.warning("Bandas no encontradas en %s: %s", safe_path.name, sorted(missing))

    return found


def safe_to_geotiff(
    safe_path: Path,
    output_dir: Path,
    spectral_bands: list[str] | None = None,
) -> Path:
    """Apila las bandas espectrales de una escena .SAFE en un GeoTIFF multibanda.

    Las bandas de 20 m (B11, B12) se remuestrean bilinearmente a 10 m.
    La banda SCL se exporta como archivo separado ``<stem>_SCL.tif``.

    Si el GeoTIFF de salida ya existe, lo retorna directamente sin reprocesar.

    Args:
        safe_path: Ruta a la carpeta ``.SAFE`` descargada.
        output_dir: Carpeta donde se guardan los GeoTIFF resultantes.
        spectral_bands: Orden de bandas en el GeoTIFF de salida.
                        Por defecto: ``["B02", "B03", "B04", "B08", "B11", "B12"]``.

    Returns:
        :class:`pathlib.Path` al GeoTIFF multibanda generado.

    Raises:
        FileNotFoundError: Si la carpeta .SAFE o la banda de referencia no se encuentran.
        RuntimeError: Si se encuentran menos de 4 bandas espectrales.
        ImportError: Si rasterio no está instalado.
    """
    try:
        import rasterio
        from rasterio.enums import Resampling
    except ImportError as exc:
        raise ImportError(
            "La librería 'rasterio' no está instalada. "
            "Ejecutá: pip install rasterio"
        ) from exc

    if spectral_bands is None:
        spectral_bands = _DEFAULT_SPECTRAL

    safe_path = Path(safe_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / (safe_path.stem + "_stacked.tif")
    if out_path.exists():
        log.info("GeoTIFF ya existe, omitiendo conversión: %s", out_path.name)
        return out_path

    log.info("Convirtiendo .SAFE → GeoTIFF multibanda: %s", safe_path.name)

    all_bands_needed = spectral_bands + ["SCL"]
    band_files = find_band_files(safe_path, all_bands_needed)

    available_spectral = [b for b in spectral_bands if b in band_files]
    if len(available_spectral) < 4:
        raise RuntimeError(
            f"No se encontraron suficientes bandas espectrales en {safe_path.name}.\n"
            f"Encontradas: {sorted(band_files.keys())}. Se necesitan al menos 4."
        )

    # --- Determinar grid de referencia (primer banda 10m disponible) ---
    ref_band = next(
        (b for b in ["B02", "B03", "B04", "B08"] if b in band_files), None
    )
    if ref_band is None:
        raise FileNotFoundError(
            "No se encontró ninguna banda de 10 m (B02/B03/B04/B08) en la escena."
        )

    with rasterio.open(band_files[ref_band]) as ref:
        ref_profile = ref.profile.copy()
        ref_width = ref.width
        ref_height = ref.height

    # --- Escribir GeoTIFF principal ---
    ref_profile.update(
        driver="GTiff",
        count=len(available_spectral),
        dtype="uint16",
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        interleave="band",
    )

    with rasterio.open(out_path, "w", **ref_profile) as dst:
        for channel_idx, band_name in enumerate(available_spectral, start=1):
            with rasterio.open(band_files[band_name]) as src:
                if src.width == ref_width and src.height == ref_height:
                    data = src.read(1)
                else:
                    # Remuestrear banda 20m → 10m
                    data = src.read(
                        1,
                        out_shape=(ref_height, ref_width),
                        resampling=Resampling.bilinear,
                    )
            dst.write(data, channel_idx)
            log.debug("Canal %d ← banda %s", channel_idx, band_name)

    log.info("GeoTIFF multibanda escrito: %s (%d bandas)", out_path.name, len(available_spectral))

    # --- SCL separado (si existe) ---
    if "SCL" in band_files:
        scl_out = output_dir / (safe_path.stem + "_SCL.tif")
        if not scl_out.exists():
            with rasterio.open(band_files["SCL"]) as scl_src:
                scl_profile = scl_src.profile.copy()
                scl_profile.update(driver="GTiff", compress="lzw")
                with rasterio.open(scl_out, "w", **scl_profile) as scl_dst:
                    scl_dst.write(scl_src.read())
            log.info("SCL exportado: %s", scl_out.name)

    return out_path
