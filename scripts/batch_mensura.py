"""Mensura masiva de techos por parcela catastral.

Pipeline end-to-end:
  1. Carga una imagen Sentinel-2 GeoTIFF (ya descargada).
  2. Obtiene las parcelas catastrales del área (desde archivo local, WFS o escaneo WMS).
  3. Ejecuta la detección de techos (motor clásico o U-Net).
  4. Intersecta los polígonos de techos con las parcelas.
  5. Exporta un CSV con una fila por parcela: área de techo, % cubierto, etc.

────────────────────────────────────────────────────
 Caso de uso principal: partido de Luján completo
────────────────────────────────────────────────────

Paso 0 — Descargar imagen Sentinel-2 (si no la tenés):
  Usá la GUI → botón "Descargar imagen S2" o el diálogo de descarga.
  Resultado: data/cache/S2A_..._stacked.tif

Paso 1 — Obtener catastro de parcelas (UNA SOLA VEZ):
  Descargá el catastro del partido de Luján desde:
    https://datos.gba.gob.ar/dataset/catastro-territorial
  Guardalo en: data/catastro/lujan.gpkg

Paso 2 — Ejecutar la mensura masiva:
  python scripts/batch_mensura.py \\
      --image data/cache/S2A_MSIL2A_20240301_stacked.tif \\
      --parcelas data/catastro/lujan.gpkg \\
      --output data/output/mensura_lujan.csv

────────────────────────────────────────────────────
 Flujo alternativo: lista de partidas
────────────────────────────────────────────────────

  Si tenés un CSV con nomenclaturas de parcelas específicas:

  python scripts/batch_mensura.py \\
      --image data/cache/S2A_..._stacked.tif \\
      --parcelas data/catastro/lujan.gpkg \\
      --partidas data/mis_partidas.csv \\
      --output data/output/mensura_parcelas.csv

  Formato de data/mis_partidas.csv (solo una columna, sin encabezado
  obligatorio, pero se detecta automáticamente):
    nomenclatura
    067-A-1-23
    067-A-1-24
    067-B-2-15

────────────────────────────────────────────────────
 Sin archivo catastral local
────────────────────────────────────────────────────

  Si no tenés el catastro descargado, el script intentará obtener
  las parcelas via WFS automáticamente. Si el servidor no responde,
  caerá en un escaneo WMS (lento) o pedirá descargar el catastro.

  python scripts/batch_mensura.py \\
      --image data/cache/S2A_..._stacked.tif \\
      --bbox "-59.15,-34.70,-58.90,-34.45" \\
      --output data/output/mensura_lujan.csv
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch_mensura")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Inputs
    p.add_argument(
        "--image", required=True, type=Path,
        help="GeoTIFF Sentinel-2 multibanda (_stacked.tif) sobre el área de interés.",
    )
    p.add_argument(
        "--parcelas", type=Path, default=None,
        help=(
            "Archivo catastral con polígonos de parcelas (.gpkg, .shp, .geojson). "
            "Recomendado para el partido completo. "
            "Descargable desde https://datos.gba.gob.ar/dataset/catastro-territorial"
        ),
    )
    p.add_argument(
        "--partidas", type=Path, default=None,
        help=(
            "CSV con nomenclaturas/partidas a procesar (una por línea). "
            "Si se omite, se procesan TODAS las parcelas del área de la imagen."
        ),
    )
    p.add_argument(
        "--bbox", type=str, default=None,
        help=(
            "Bounding box WGS84 para obtener parcelas: "
            "'lon_min,lat_min,lon_max,lat_max'. "
            "Si se omite, se usa el extent de la imagen automáticamente."
        ),
    )
    p.add_argument(
        "--wfs-url", type=str, default=None,
        help="URL WFS personalizada para obtener parcelas (opcional).",
    )

    # Detección
    p.add_argument(
        "--engine", choices=["clasico", "unet"], default="clasico",
        help="Motor de detección de techos (default: clasico).",
    )
    p.add_argument(
        "--model", type=Path, default=None,
        help="Ruta a pesos U-Net (.pth). Solo se usa con --engine unet.",
    )

    # Output
    p.add_argument(
        "--output", type=Path, default=Path("data/output/mensura.csv"),
        help="Ruta del CSV de salida (default: data/output/mensura.csv).",
    )
    p.add_argument(
        "--output-geojson", action="store_true",
        help="Además del CSV, exportar el resultado como GeoJSON con geometrías.",
    )
    p.add_argument(
        "--include-coords", action="store_true",
        help="Incluir columnas centroide_lat, centroide_lon en el CSV.",
    )
    p.add_argument(
        "--min-roof-m2", type=float, default=10.0,
        help="Superficie mínima en m² para considerar un polígono como techo (default: 10.0).",
    )
    p.add_argument(
        "--scan-step", type=float, default=0.003,
        help=(
            "Paso de grilla en grados para el escaneo WMS fallback "
            "(default: 0.003° ≈ 330 m). Solo aplica si no hay --parcelas ni WFS."
        ),
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_bbox(bbox_str: str) -> tuple[float, float, float, float]:
    parts = [x.strip() for x in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"El bbox debe tener 4 valores: lon_min,lat_min,lon_max,lat_max. Recibido: {bbox_str!r}")
    return tuple(float(p) for p in parts)


def _load_partidas(path: Path) -> list[str]:
    """Lee un CSV de nomenclaturas (una por línea, ignora encabezado si no es numérico)."""
    partidas = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            val = line.strip()
            if not val:
                continue
            # Ignorar encabezado si la primera línea parece texto
            if i == 0 and not any(ch.isdigit() for ch in val):
                log.info("Ignorando encabezado del CSV de partidas: %r", val)
                continue
            partidas.append(val)
    log.info("Partidas cargadas desde %s: %d", path.name, len(partidas))
    return partidas


def _bbox_from_image(tif_path: Path) -> tuple[float, float, float, float]:
    """Extrae el bbox WGS84 de un GeoTIFF."""
    try:
        import rasterio
        from pyproj import Transformer
        with rasterio.open(tif_path) as src:
            crs = src.crs
            bounds = src.bounds
        if "4326" in str(crs):
            return (bounds.left, bounds.bottom, bounds.right, bounds.top)
        t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon_min, lat_min = t.transform(bounds.left, bounds.bottom)
        lon_max, lat_max = t.transform(bounds.right, bounds.top)
        return (lon_min, lat_min, lon_max, lat_max)
    except Exception as exc:
        raise RuntimeError(f"No se pudo obtener el bbox de la imagen: {exc}") from exc


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------
    # Validar inputs
    # ------------------------------------------------------------------
    if not args.image.exists():
        log.error("Imagen no encontrada: %s", args.image)
        sys.exit(1)

    partidas: list[str] | None = None
    if args.partidas is not None:
        if not args.partidas.exists():
            log.error("Archivo de partidas no encontrado: %s", args.partidas)
            sys.exit(1)
        partidas = _load_partidas(args.partidas)
        if not partidas:
            log.error("El archivo de partidas está vacío.")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Determinar bbox de trabajo
    # ------------------------------------------------------------------
    if args.bbox:
        try:
            bbox = _parse_bbox(args.bbox)
        except ValueError as exc:
            log.error("Error en --bbox: %s", exc)
            sys.exit(1)
    else:
        log.info("Calculando bbox automático desde la imagen…")
        try:
            bbox = _bbox_from_image(args.image)
        except RuntimeError as exc:
            log.error("%s", exc)
            sys.exit(1)

    log.info("Bbox de trabajo: lon_min=%.5f lat_min=%.5f lon_max=%.5f lat_max=%.5f", *bbox)

    # ------------------------------------------------------------------
    # Paso 1: Cargar imagen
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("PASO 1 — Cargando imagen Sentinel-2: %s", args.image.name)

    try:
        from roofscan.core.ingesta.loader import load_geotiff
        data = load_geotiff(args.image)
    except Exception as exc:
        log.error("No se pudo cargar la imagen: %s", exc)
        sys.exit(1)

    log.info(
        "  Imagen cargada | bandas=%d | resolución=%.1f m/px | CRS=%s",
        data["count"], data.get("resolution_m") or 0, data["crs"],
    )

    # ------------------------------------------------------------------
    # Paso 2: Obtener parcelas
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("PASO 2 — Obteniendo parcelas catastrales…")

    try:
        from roofscan.core.ingesta.wfs_arba import get_parcelas
        gdf_parcelas = get_parcelas(
            bbox_wgs84=bbox,
            local_file=args.parcelas,
            wfs_url=args.wfs_url,
            nomenclaturas=partidas,
            scan_step_deg=args.scan_step,
        )
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        log.error("Error al obtener parcelas: %s", exc, exc_info=True)
        sys.exit(1)

    log.info("  Parcelas obtenidas: %d", len(gdf_parcelas))

    # ------------------------------------------------------------------
    # Paso 3: Detección de techos
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("PASO 3 — Detección de techos (motor: %s)…", args.engine)

    array = data["array"].astype("float32")

    try:
        if args.engine == "clasico":
            from roofscan.core.deteccion.clasico.spectral_indices import detect_roofs
            det_result = detect_roofs(array)
            mask_raw = det_result["mask"]

        else:  # unet
            from roofscan.core.deteccion.dl.unet import build_unet, load_weights
            from roofscan.core.deteccion.dl.predictor import predict_mask
            from roofscan.config import MODELS_DIR

            model_path = args.model or MODELS_DIR / "unet_best.pth"
            if not model_path.exists():
                log.error(
                    "No se encontraron pesos U-Net en: %s\n"
                    "  Entrenamiento previo requerido. Usá --engine clasico o "
                    "ejecutá primero scripts/pretrain_unet.py",
                    model_path,
                )
                sys.exit(1)

            model = build_unet(pretrained=False)
            model = load_weights(model, model_path)
            model.eval()

            # Normalizar S2 DN → [0, 1] si aún no está normalizado
            if array.max() > 1.0:
                array = (array / 10_000.0).clip(0.0, 1.0)

            mask_raw = predict_mask(model, array)

    except Exception as exc:
        log.error("Error en la detección: %s", exc, exc_info=True)
        sys.exit(1)

    # Post-proceso morfológico
    try:
        from roofscan.core.deteccion.clasico.morphology import run_morphology
        morph = run_morphology(mask_raw)
        labels = morph["labels"]
        n_roofs = morph["n_roofs"]
        log.info("  Techos detectados: %d objetos", n_roofs)
    except Exception as exc:
        log.error("Error en morfología: %s", exc, exc_info=True)
        sys.exit(1)

    if n_roofs == 0:
        log.warning(
            "No se detectaron techos en la imagen.\n"
            "  Si usás motor clásico, probá con imágenes de menor nubosidad.\n"
            "  Si usás U-Net, verificá que los pesos sean correctos."
        )

    # Calcular áreas y vectorizar
    try:
        from roofscan.core.calculo.area_calculator import calculate_areas
        from roofscan.core.calculo.geometry_merger import labels_to_geodataframe

        resolution_m = data.get("resolution_m") or 10.0
        areas = calculate_areas(labels, resolution_m, min_area_m2=args.min_roof_m2)
        log.info(
            "  Área total de techos: %.1f m² (%.2f ha)",
            sum(a["area_m2"] for a in areas),
            sum(a["area_m2"] for a in areas) / 10_000,
        )

        gdf_roofs = labels_to_geodataframe(
            labels, data["transform"], data["crs"], areas=areas
        )
        log.info("  Polígonos de techos vectorizados: %d", len(gdf_roofs))

    except Exception as exc:
        log.error("Error en vectorización: %s", exc, exc_info=True)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Paso 4: Intersección espacial techo ↔ parcela
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("PASO 4 — Cruzando techos con parcelas (spatial join)…")

    try:
        from roofscan.core.calculo.spatial_join import (
            join_roofs_to_parcelas, export_mensura_csv, summary_stats
        )
        resultado = join_roofs_to_parcelas(gdf_roofs, gdf_parcelas)
    except Exception as exc:
        log.error("Error en la intersección espacial: %s", exc, exc_info=True)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Paso 5: Exportar resultados
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("PASO 5 — Exportando resultados…")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        export_mensura_csv(resultado, args.output, include_geometry=args.include_coords)
    except Exception as exc:
        log.error("Error al exportar CSV: %s", exc, exc_info=True)
        sys.exit(1)

    if args.output_geojson:
        geojson_path = args.output.with_suffix(".geojson")
        try:
            resultado.to_file(geojson_path, driver="GeoJSON")
            log.info("GeoJSON exportado: %s", geojson_path)
        except Exception as exc:
            log.warning("No se pudo exportar GeoJSON: %s", exc)

    # ------------------------------------------------------------------
    # Resumen final
    # ------------------------------------------------------------------
    stats = summary_stats(resultado)
    log.info("=" * 60)
    log.info("MENSURA COMPLETA")
    log.info("  Parcelas procesadas:     %d", stats["total_parcelas"])
    log.info("  Parcelas con techo:      %d", stats["parcelas_con_techo"])
    log.info("  Parcelas sin techo:      %d", stats["parcelas_sin_techo"])
    log.info("  Área total techos:       %.1f m² (%.2f ha)",
             stats["area_total_techos_m2"], stats["area_total_techos_m2"] / 10_000)
    log.info("  Área media por parcela:  %.1f m²", stats["area_media_techo_m2"])
    log.info("  Cobertura media:         %.1f%%", stats["pct_cubierto_medio"])
    log.info("  CSV generado:            %s", args.output.resolve())
    log.info("=" * 60)


if __name__ == "__main__":
    main()
