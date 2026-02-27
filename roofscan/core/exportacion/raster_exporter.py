"""Exportación de rasters procesados a GeoTIFF y PNG de previsualización.

Permite guardar el array preprocesado como GeoTIFF georreferenciado (para
usar en QGIS u otros SIG) y como PNG RGB para revisión visual rápida.

Uso típico::

    from roofscan.core.exportacion.raster_exporter import export_geotiff, export_preview_png

    # Después del pipeline de preprocesamiento:
    tif_path = export_geotiff(result, output_dir, filename="lujan_s2_preprocesado")
    png_path = export_preview_png(result, output_dir, rgb_bands=(3, 2, 1))
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def export_geotiff(
    data: dict[str, Any],
    output_dir: Path | str,
    filename: str = "raster_procesado",
    dtype: str = "float32",
) -> Path:
    """Exporta el array de un dict de raster a un archivo GeoTIFF georreferenciado.

    Args:
        data: Dict con claves ``array``, ``crs``, ``transform`` y opcionalmente
              ``nodata``. Compatible con la salida del pipeline de preprocesamiento.
        output_dir: Directorio donde se guardará el archivo.
        filename: Nombre del archivo sin extensión.
        dtype: Tipo de dato del GeoTIFF de salida (ej. ``"float32"``, ``"uint16"``).

    Returns:
        :class:`pathlib.Path` al archivo ``.tif`` creado.

    Raises:
        ImportError: Si rasterio no está instalado.
        ValueError: Si el dict no contiene los campos requeridos.
    """
    try:
        import rasterio
        from rasterio.crs import CRS
    except ImportError as exc:
        raise ImportError("Instalar rasterio: pip install rasterio") from exc

    _validate_data_keys(data, required=("array", "crs", "transform"))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{filename}.tif"

    array = data["array"]
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    n_bands, height, width = array.shape

    crs = CRS.from_user_input(data["crs"])
    nodata = data.get("nodata", None)

    # Para float32, usar NaN como nodata si no hay uno explícito
    if nodata is None and np.issubdtype(array.dtype, np.floating):
        nodata = np.nan

    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=n_bands,
        dtype=dtype,
        crs=crs,
        transform=data["transform"],
        nodata=nodata,
        compress="lzw",  # compresión sin pérdida
    ) as dst:
        dst.write(array.astype(dtype))

    size_mb = filepath.stat().st_size / 1e6
    log.info("GeoTIFF exportado: %s (%.1f MB)", filepath, size_mb)
    return filepath


def export_preview_png(
    data: dict[str, Any],
    output_dir: Path | str,
    filename: str = "preview",
    rgb_bands: tuple[int, int, int] = (3, 2, 1),
) -> Path:
    """Exporta una previsualización RGB como PNG para revisión visual rápida.

    La imagen resultante NO está georreferenciada (es solo para visualización).
    Para análisis SIG usar :func:`export_geotiff`.

    Args:
        data: Dict con clave ``array`` (bandas, alto, ancho) en rango [0.0, 1.0].
        output_dir: Directorio de destino.
        filename: Nombre del archivo sin extensión.
        rgb_bands: Índices 1-based de las bandas a usar como R, G, B.
                   Por defecto ``(3, 2, 1)`` = bandas B04, B03, B02 de Sentinel-2
                   (True color RGB).

    Returns:
        :class:`pathlib.Path` al archivo ``.png`` creado.

    Raises:
        ImportError: Si matplotlib no está instalado.
        ValueError: Si los índices de banda son inválidos.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # sin ventana de display (headless)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Instalar matplotlib: pip install matplotlib") from exc

    _validate_data_keys(data, required=("array",))

    array = data["array"]
    n_bands = array.shape[0]

    # Validar índices de banda
    for i, b in enumerate(rgb_bands, 1):
        if not (1 <= b <= n_bands):
            raise ValueError(
                f"Índice de banda {b} inválido (el raster tiene {n_bands} bandas). "
                "Los índices son 1-based."
            )

    r, g, b_ch = rgb_bands
    rgb = np.stack([
        array[r - 1],
        array[g - 1],
        array[b_ch - 1],
    ], axis=-1)  # (H, W, 3)

    # Normalizar a [0, 1] reemplazando NaN por 0 (negro en la visualización)
    rgb = np.nan_to_num(rgb, nan=0.0)
    rgb_min, rgb_max = rgb.min(), rgb.max()
    if rgb_max > rgb_min:
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
    rgb = np.clip(rgb, 0.0, 1.0)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{filename}.png"

    dpi = 150
    fig_w = max(4, rgb.shape[1] / dpi)
    fig_h = max(3, rgb.shape[0] / dpi)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(rgb)
    ax.set_title(f"Preview RGB (B{r}, B{g}, B{b_ch})", fontsize=8)
    ax.axis("off")
    plt.tight_layout(pad=0.2)
    plt.savefig(filepath, bbox_inches="tight", dpi=dpi)
    plt.close(fig)

    log.info("PNG de previsualización exportado: %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# Helper privado
# ---------------------------------------------------------------------------

def _validate_data_keys(data: dict, required: tuple[str, ...]) -> None:
    """Verifica que el dict contenga las claves requeridas."""
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"El dict de datos no contiene las claves requeridas: {missing}"
        )
