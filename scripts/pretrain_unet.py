"""Pre-entrenamiento del modelo U-Net con tiles de edificios.

Entrena el modelo U-Net (ResNet34 + pesos ImageNet) sobre el dataset de
tiles generado por ``prepare_tiles.py``, produciendo un archivo de pesos
``.pth`` listo para usar en RoofScan sin necesidad de datos locales etiquetados.

────────────────────────────────────────────────────
 Flujo completo
────────────────────────────────────────────────────
1. Generar tiles de entrenamiento::

    python scripts/prepare_tiles.py \\
        --buildings data/pretrain/ARG.gpkg \\
        --tiles-dir data/cache/ \\
        --output-dir data/pretrain/

2. Pre-entrenar el modelo (este script)::

    python scripts/pretrain_unet.py \\
        --tiles-dir data/pretrain/ \\
        --output data/models/unet_pretrained.pth

3. La aplicación cargará automáticamente los pesos si el archivo existe
   en la ruta configurada en ``roofscan/config.py`` (``MODELS_DIR``).

────────────────────────────────────────────────────
 Requisitos de hardware
────────────────────────────────────────────────────
- CPU: entrenamiento lento (~10 min/época para 500 tiles).
- GPU NVIDIA: 10–20× más rápido. Detectada automáticamente si está
  disponible (requiere CUDA).

────────────────────────────────────────────────────
 Parámetros recomendados
────────────────────────────────────────────────────
- Dataset pequeño  (< 200 tiles): --epochs 30 --lr 1e-4
- Dataset mediano  (200–1000 tiles): --epochs 50 --lr 1e-4
- Dataset grande   (> 1000 tiles): --epochs 80 --lr 5e-5
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
log = logging.getLogger("pretrain_unet")

# ---------------------------------------------------------------------------
# Valores por defecto
# ---------------------------------------------------------------------------
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE = 4
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_PATIENCE = 8


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--tiles-dir", required=True, type=Path,
        help="Directorio con las subcarpetas images/ y masks/ generadas por prepare_tiles.py",
    )
    p.add_argument(
        "--output", type=Path, default=Path("data/models/unet_pretrained.pth"),
        help="Ruta de salida para los pesos del modelo (default: data/models/unet_pretrained.pth)",
    )
    p.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Número máximo de épocas de entrenamiento (default: {DEFAULT_EPOCHS})",
    )
    p.add_argument(
        "--lr", type=float, default=DEFAULT_LR,
        help=f"Tasa de aprendizaje para Adam (default: {DEFAULT_LR})",
    )
    p.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Tamaño de lote (default: {DEFAULT_BATCH_SIZE})",
    )
    p.add_argument(
        "--val-split", type=float, default=DEFAULT_VAL_SPLIT,
        help=f"Fracción del dataset para validación (default: {DEFAULT_VAL_SPLIT})",
    )
    p.add_argument(
        "--patience", type=int, default=DEFAULT_PATIENCE,
        help=f"Épocas sin mejora antes de early stopping (default: {DEFAULT_PATIENCE})",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Dispositivo PyTorch: 'cpu' o 'cuda'. Detectado automáticamente si no se especifica.",
    )
    p.add_argument(
        "--no-pretrained", action="store_true",
        help="Inicializar el encoder sin pesos ImageNet (más lento en converger, no recomendado).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Validación del dataset
# ---------------------------------------------------------------------------

def _validate_dataset(tiles_dir: Path) -> int:
    """Verifica que el directorio de tiles tenga la estructura correcta.

    Returns:
        Número de pares (imagen, máscara) encontrados.
    """
    img_dir = tiles_dir / "images"
    msk_dir = tiles_dir / "masks"

    if not img_dir.is_dir():
        log.error(
            "No se encontró el directorio de imágenes: %s\n"
            "  Ejecutá primero: python scripts/prepare_tiles.py",
            img_dir,
        )
        sys.exit(1)

    if not msk_dir.is_dir():
        log.error(
            "No se encontró el directorio de máscaras: %s\n"
            "  Ejecutá primero: python scripts/prepare_tiles.py",
            msk_dir,
        )
        sys.exit(1)

    img_names = {p.name for p in img_dir.glob("*.npy")}
    msk_names = {p.name for p in msk_dir.glob("*.npy")}
    common = img_names & msk_names

    if not common:
        log.error(
            "No se encontraron pares imagen/máscara en %s.\n"
            "  Verificá que las carpetas images/ y masks/ contengan archivos .npy\n"
            "  con el mismo nombre de archivo.",
            tiles_dir,
        )
        sys.exit(1)

    only_imgs = img_names - msk_names
    only_msks = msk_names - img_names
    if only_imgs:
        log.warning("Imágenes sin máscara correspondiente (%d): se ignorarán.", len(only_imgs))
    if only_msks:
        log.warning("Máscaras sin imagen correspondiente (%d): se ignorarán.", len(only_msks))

    n = len(common)
    log.info("Dataset validado: %d pares imagen/máscara encontrados.", n)

    if n < 10:
        log.warning(
            "Dataset muy pequeño (%d tiles). El modelo puede no generalizar bien.\n"
            "  Se recomienda ≥ 50 tiles. Considerá:\n"
            "  - Agregar más GeoTIFFs Sentinel-2.\n"
            "  - Reducir --min-roof-pct en prepare_tiles.py.",
            n,
        )
    elif n < 50:
        log.warning(
            "Dataset pequeño (%d tiles). Para mejores resultados se recomiendan ≥ 50.", n
        )

    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Validar dataset
    n_tiles = _validate_dataset(args.tiles_dir)

    # Crear directorio de salida
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Importar dependencias del proyecto
    try:
        from roofscan.core.deteccion.dl.unet import build_unet
        from roofscan.core.deteccion.dl.trainer import fine_tune
    except ImportError as exc:
        log.error(
            "No se pudo importar el módulo roofscan: %s\n"
            "  Asegurate de ejecutar este script desde la raíz del proyecto:\n"
            "  python scripts/pretrain_unet.py --tiles-dir ...",
            exc,
        )
        sys.exit(1)

    # Construir modelo
    log.info("=" * 60)
    log.info("Construyendo U-Net (encoder=resnet34, pretrained=%s)...", not args.no_pretrained)
    try:
        model = build_unet(pretrained=not args.no_pretrained, device=args.device)
    except ImportError as exc:
        log.error("Dependencia faltante: %s", exc)
        sys.exit(1)

    # Entrenamiento
    log.info("=" * 60)
    log.info("Iniciando pre-entrenamiento:")
    log.info("  Dataset:    %s (%d tiles)", args.tiles_dir, n_tiles)
    log.info("  Épocas:     %d (patience=%d)", args.epochs, args.patience)
    log.info("  LR:         %.0e", args.lr)
    log.info("  Batch size: %d", args.batch_size)
    log.info("  Val split:  %.0f%%", args.val_split * 100)
    log.info("  Salida:     %s", args.output.resolve())
    log.info("=" * 60)

    try:
        history = fine_tune(
            model=model,
            dataset_dir=args.tiles_dir,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            val_split=args.val_split,
            device=args.device,
            save_path=args.output,
            patience=args.patience,
        )
    except (FileNotFoundError, ValueError) as exc:
        log.error("Error en el entrenamiento: %s", exc)
        sys.exit(1)
    except Exception as exc:
        log.error("Error inesperado durante el entrenamiento: %s", exc, exc_info=True)
        sys.exit(1)

    # Resumen final
    log.info("=" * 60)
    log.info("Pre-entrenamiento completo.")
    log.info("  Épocas ejecutadas: %d", len(history["train_loss"]))
    log.info("  Mejor época:       %d", history["best_epoch"])
    log.info("  Mejor val_loss:    %.4f", history["best_val_loss"])
    log.info("  Pesos guardados:   %s", args.output.resolve())

    if args.output.exists():
        size_mb = args.output.stat().st_size / 1e6
        log.info("  Tamaño del archivo: %.1f MB", size_mb)

    log.info("=" * 60)
    log.info(
        "Siguiente paso: cargá los pesos en la aplicación con:\n"
        "  from roofscan.core.deteccion.dl.unet import build_unet, load_weights\n"
        "  model = build_unet(pretrained=False)\n"
        "  model = load_weights(model, '%s')",
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
