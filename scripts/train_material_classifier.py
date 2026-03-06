"""Entrenamiento supervisado del clasificador de material de techo.

Entrena el MLP espectral y/o la CNN multi-escala definidos en
``material_classifier.py`` usando techos etiquetados manualmente.

────────────────────────────────────────────────────
 Workflow completo
────────────────────────────────────────────────────
1. Ejecutar detección de techos desde la GUI (Motor Clásico o U-Net).
2. Exportar GeoJSON de techos detectados (botón Exportar GeoJSON).
3. Abrir el GeoJSON en QGIS y agregar una columna ``material`` con uno
   de los siguientes valores (exactos, case-sensitive):
       - ``zinc_corrugado``
       - ``losa_hormigon``
       - ``tejas_ceramica``
       - ``construccion_incompleta``
4. Inspeccionar cada techo visualmente en Google Maps Satellite para
   determinar el material correcto.
5. Guardar como ``data/pretrain/lujan_techos_etiquetados.geojson``.
6. Ejecutar este script::

    python scripts/train_material_classifier.py \\
        --image data/cache/lujan_YYYYMMDD_stacked.tif \\
        --labels data/pretrain/lujan_techos_etiquetados.geojson

────────────────────────────────────────────────────
 Salidas
────────────────────────────────────────────────────
- ``data/models/material_mlp.pth``  (MLP espectral — ligero, 29 features)
- ``data/models/material_cnn.pth``  (CNN multi-escala — requiere ≥ 50 muestras
                                     y parches 32×32 px válidos)

────────────────────────────────────────────────────
 Requisitos de datos
────────────────────────────────────────────────────
- Mínimo recomendado: 20 techos etiquetados.
- Mínimo para CNN:    50 techos con parche 32×32 válido (centroide dentro
                      de la imagen y cobertura suficiente).
- Distribución equilibrada (aprox.): ≥ 5 muestras por clase.
  Con pocas muestras se aplica sobremuestreo automático (WeightedRandomSampler).

────────────────────────────────────────────────────
 Requisitos de hardware
────────────────────────────────────────────────────
- CPU: suficiente para ambos modelos (MLP < 30 s, CNN < 5 min para 200 muestras).
- GPU NVIDIA: 10–20× más rápido para la CNN.

────────────────────────────────────────────────────
 Parámetros recomendados
────────────────────────────────────────────────────
- Dataset pequeño  (< 50 muestras):   --epochs 80  --lr 1e-3
- Dataset mediano  (50–200 muestras): --epochs 100 --lr 5e-4
- Dataset grande   (> 200 muestras):  --epochs 150 --lr 1e-4
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
log = logging.getLogger("train_material")

# ---------------------------------------------------------------------------
# Valores por defecto
# ---------------------------------------------------------------------------

DEFAULT_EPOCHS = 100
DEFAULT_LR = 5e-4
DEFAULT_BATCH_SIZE = 16
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_PATIENCE = 15
DEFAULT_PATCH_SIZE = 32
MIN_SAMPLES_TOTAL = 10
MIN_SAMPLES_CNN = 30


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--image", required=True, type=Path,
        help="GeoTIFF Sentinel-2 normalizado con 6 bandas (salida de normalize_s2).",
    )
    p.add_argument(
        "--labels", required=True, type=Path,
        help="GeoJSON/GPKG con polígonos de techos y columna 'material' etiquetada.",
    )
    p.add_argument(
        "--output-mlp", type=Path, default=Path("data/models/material_mlp.pth"),
        help="Ruta de salida para los pesos del MLP (default: data/models/material_mlp.pth).",
    )
    p.add_argument(
        "--output-cnn", type=Path, default=Path("data/models/material_cnn.pth"),
        help="Ruta de salida para los pesos de la CNN (default: data/models/material_cnn.pth).",
    )
    p.add_argument(
        "--no-mlp", action="store_true",
        help="Omitir entrenamiento del MLP espectral.",
    )
    p.add_argument(
        "--no-cnn", action="store_true",
        help="Omitir entrenamiento de la CNN multi-escala.",
    )
    p.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Número máximo de épocas (default: {DEFAULT_EPOCHS}).",
    )
    p.add_argument(
        "--lr", type=float, default=DEFAULT_LR,
        help=f"Tasa de aprendizaje para Adam (default: {DEFAULT_LR}).",
    )
    p.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Tamaño de lote (default: {DEFAULT_BATCH_SIZE}).",
    )
    p.add_argument(
        "--val-split", type=float, default=DEFAULT_VAL_SPLIT,
        help=f"Fracción del dataset para validación (default: {DEFAULT_VAL_SPLIT}).",
    )
    p.add_argument(
        "--patience", type=int, default=DEFAULT_PATIENCE,
        help=f"Épocas sin mejora antes de early stopping (default: {DEFAULT_PATIENCE}).",
    )
    p.add_argument(
        "--patch-size", type=int, default=DEFAULT_PATCH_SIZE,
        help=f"Tamaño del parche para la CNN en píxeles (default: {DEFAULT_PATCH_SIZE}).",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Dispositivo PyTorch: 'cpu' o 'cuda'. Detectado automáticamente si no se especifica.",
    )
    p.add_argument(
        "--label-col", type=str, default="material",
        help="Nombre de la columna con las etiquetas de material (default: 'material').",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Carga y validación de datos
# ---------------------------------------------------------------------------

def _load_image(image_path: Path) -> dict:
    """Carga el GeoTIFF Sentinel-2 y valida su formato."""
    try:
        from roofscan.core.ingesta.loader import load_geotiff
    except ImportError as exc:
        log.error(
            "No se pudo importar roofscan: %s\n"
            "  Ejecutá este script desde la raíz del proyecto.",
            exc,
        )
        sys.exit(1)

    log.info("Cargando imagen: %s", image_path)
    if not image_path.exists():
        log.error("El archivo de imagen no existe: %s", image_path)
        sys.exit(1)

    try:
        data = load_geotiff(str(image_path))
    except Exception as exc:
        log.error("Error al cargar la imagen: %s", exc)
        sys.exit(1)

    arr = data["array"]
    if arr.ndim != 3 or arr.shape[0] != 6:
        log.error(
            "La imagen debe tener 6 bandas (B02-B03-B04-B08-B11-B12). "
            "Bandas encontradas: %d. ¿Aplicaste normalize_s2()?",
            arr.shape[0] if arr.ndim == 3 else -1,
        )
        sys.exit(1)

    log.info(
        "  Imagen cargada: %d×%d px | %d bandas | CRS=%s | res=%.1f m/px",
        arr.shape[2], arr.shape[1], arr.shape[0],
        data.get("crs", "desconocido"),
        data.get("resolution_m", -1),
    )

    # Normalizar a float32 [0, 1] si el array está en uint (no normalizado aún)
    if arr.dtype.kind == "u":
        log.warning(
            "Array en formato entero (%s). Normalizando a [0, 1] con max=%d.",
            arr.dtype, arr.max(),
        )
        arr_max = arr.max()
        if arr_max > 0:
            data["array"] = (arr.astype("float32") / arr_max)
        else:
            log.error("El array tiene máximo == 0. Verificá la imagen.")
            sys.exit(1)

    return data


def _load_labels(labels_path: Path, label_col: str, image_crs: str) -> "geopandas.GeoDataFrame":  # noqa: F821
    """Carga el GeoJSON etiquetado y lo reproyecta al CRS de la imagen."""
    try:
        import geopandas as gpd
    except ImportError:
        log.error("geopandas no está instalado. Instalalo con: pip install geopandas")
        sys.exit(1)

    log.info("Cargando etiquetas: %s", labels_path)
    if not labels_path.exists():
        log.error("El archivo de etiquetas no existe: %s", labels_path)
        sys.exit(1)

    try:
        gdf = gpd.read_file(str(labels_path))
    except Exception as exc:
        log.error("Error al cargar el archivo de etiquetas: %s", exc)
        sys.exit(1)

    if label_col not in gdf.columns:
        log.error(
            "La columna '%s' no existe en el archivo de etiquetas.\n"
            "  Columnas disponibles: %s\n"
            "  Usá --label-col para especificar la columna correcta.",
            label_col, list(gdf.columns),
        )
        sys.exit(1)

    # Reproyectar al CRS de la imagen
    if gdf.crs is not None and str(gdf.crs) != image_crs:
        log.info("  Reproyectando etiquetas de %s → %s", gdf.crs, image_crs)
        try:
            gdf = gdf.to_crs(image_crs)
        except Exception as exc:
            log.error("Error al reproyectar etiquetas: %s", exc)
            sys.exit(1)

    log.info("  Etiquetas cargadas: %d polígonos", len(gdf))
    return gdf


def _validate_and_filter_labels(
    gdf: "geopandas.GeoDataFrame",  # noqa: F821
    label_col: str,
) -> "geopandas.GeoDataFrame":  # noqa: F821
    """Valida las etiquetas y elimina filas con materiales desconocidos."""
    from roofscan.core.deteccion.dl.material_classifier import MATERIAL_LABELS

    # Eliminar filas sin etiqueta
    n_before = len(gdf)
    gdf = gdf.dropna(subset=[label_col]).copy()
    n_dropped = n_before - len(gdf)
    if n_dropped > 0:
        log.warning("  %d filas sin etiqueta eliminadas.", n_dropped)

    # Normalizar espacios y verificar valores válidos
    gdf[label_col] = gdf[label_col].str.strip()
    invalid_mask = ~gdf[label_col].isin(MATERIAL_LABELS)
    invalid_values = gdf.loc[invalid_mask, label_col].unique().tolist()
    if invalid_values:
        log.warning(
            "  %d filas con material desconocido eliminadas: %s\n"
            "  Valores válidos: %s",
            invalid_mask.sum(), invalid_values, MATERIAL_LABELS,
        )
        gdf = gdf[~invalid_mask].copy()

    if len(gdf) == 0:
        log.error(
            "No quedan muestras válidas después de filtrar.\n"
            "  Verificá la columna '%s' en el archivo de etiquetas.",
            label_col,
        )
        sys.exit(1)

    # Distribución de clases
    log.info("  Distribución de muestras por clase:")
    for cls in MATERIAL_LABELS:
        n = (gdf[label_col] == cls).sum()
        bar = "█" * min(n, 40)
        log.info("    %-28s %3d  %s", cls, n, bar)

    return gdf


# ---------------------------------------------------------------------------
# Extracción de features
# ---------------------------------------------------------------------------

def _extract_features(
    gdf: "geopandas.GeoDataFrame",  # noqa: F821
    image_data: dict,
    label_col: str,
    patch_size: int,
) -> tuple:
    """Extrae features MLP y parches CNN para todos los techos etiquetados.

    Returns:
        Tuple ``(X_mlp, X_cnn, y_mlp, y_cnn)`` donde:
            - ``X_mlp``:  array float32 ``(N_mlp, 29)``
            - ``y_mlp``:  array int64   ``(N_mlp,)``
            - ``X_cnn``:  array float32 ``(N_cnn, 6, patch_size, patch_size)``
            - ``y_cnn``:  array int64   ``(N_cnn,)``

        Los subconjuntos MLP y CNN pueden ser de tamaño distinto porque
        algunos polígonos generan features MLP pero no parches CNN válidos
        (centroide fuera de imagen) o viceversa.
    """
    import numpy as np
    from roofscan.core.deteccion.dl.material_classifier import (
        MATERIAL_IDX,
        extract_spectral_stats,
        extract_roof_patch,
    )

    arr = image_data["array"]
    transform = image_data["transform"]

    mlp_feats, mlp_labels = [], []
    cnn_patches, cnn_labels = [], []
    n_ok_mlp = n_ok_cnn = n_skip_mlp = n_skip_cnn = 0

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            n_skip_mlp += 1
            n_skip_cnn += 1
            continue

        label_idx = MATERIAL_IDX[row[label_col]]

        # --- Features MLP ---
        feats = extract_spectral_stats(arr, transform, geom)
        if feats is not None:
            mlp_feats.append(feats)
            mlp_labels.append(label_idx)
            n_ok_mlp += 1
        else:
            n_skip_mlp += 1

        # --- Parche CNN ---
        patch = extract_roof_patch(arr, transform, geom, patch_size=patch_size)
        if patch is not None:
            cnn_patches.append(patch)
            cnn_labels.append(label_idx)
            n_ok_cnn += 1
        else:
            n_skip_cnn += 1

    X_mlp = np.stack(mlp_feats).astype(np.float32) if mlp_feats else np.empty((0, 29), np.float32)
    y_mlp = np.array(mlp_labels, dtype=np.int64)
    X_cnn = (
        np.stack(cnn_patches).astype(np.float32)
        if cnn_patches
        else np.empty((0, 6, patch_size, patch_size), np.float32)
    )
    y_cnn = np.array(cnn_labels, dtype=np.int64)

    log.info(
        "Features extraídas → MLP: %d/%d válidas | CNN: %d/%d válidas",
        n_ok_mlp, n_ok_mlp + n_skip_mlp,
        n_ok_cnn, n_ok_cnn + n_skip_cnn,
    )
    return X_mlp, y_mlp, X_cnn, y_cnn


# ---------------------------------------------------------------------------
# Utilidades de entrenamiento
# ---------------------------------------------------------------------------

def _train_val_split(X, y, val_split: float, rng_seed: int = 42):
    """Divide en train/val estratificado por clase."""
    import numpy as np

    n = len(X)
    rng = np.random.default_rng(rng_seed)
    classes = np.unique(y)
    train_idx, val_idx = [], []

    for cls in classes:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_split))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    # Si algún split quedó vacío (pocas muestras), todo va a train
    if not val_idx:
        train_idx = list(range(n))
        val_idx = list(range(n))
        log.warning("Dataset muy pequeño: se usa el mismo split para train y validación.")

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
    )


def _class_weights(y, n_classes: int):
    """Calcula pesos inversos a la frecuencia de clase."""
    import numpy as np
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)  # evitar división por cero
    weights = 1.0 / counts
    return (weights / weights.sum() * n_classes).astype(np.float32)


def _sample_weights(y, n_classes: int):
    """Peso por muestra para WeightedRandomSampler (datasets desbalanceados)."""
    import numpy as np
    cw = _class_weights(y, n_classes)
    return cw[y]


# ---------------------------------------------------------------------------
# Entrenamiento MLP
# ---------------------------------------------------------------------------

def _train_mlp(
    X_tr, y_tr, X_va, y_va,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    device: str | None,
    output_path: Path,
) -> dict:
    """Entrena el MLP espectral con early stopping.

    Returns:
        Diccionario con historial de pérdidas y métricas de la mejor época.
    """
    import numpy as np

    torch = _import_torch()
    from roofscan.core.deteccion.dl.material_classifier import (
        N_CLASSES, N_SPECTRAL_FEATURES,
        build_material_mlp, save_weights,
    )
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info("Entrenando MLP espectral en device=%s", dev)

    # Datasets
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    Xv = torch.tensor(X_va, dtype=torch.float32).to(dev)
    yv = torch.tensor(y_va, dtype=torch.long).to(dev)

    # Sobremuestreo si el dataset está desbalanceado
    sw = _sample_weights(y_tr, N_CLASSES)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sw, dtype=torch.double),
        num_samples=len(sw),
        replacement=True,
    )
    loader = DataLoader(
        TensorDataset(Xt, yt),
        batch_size=min(batch_size, len(Xt)),
        sampler=sampler,
    )

    # Pérdida ponderada para clases desbalanceadas
    cw = torch.tensor(_class_weights(y_tr, N_CLASSES), dtype=torch.float32).to(dev)
    criterion = torch.nn.CrossEntropyLoss(weight=cw)

    model = build_material_mlp(device=str(dev))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=patience // 3, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_acc = 0.0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "best_epoch": 0}

    for epoch in range(1, epochs + 1):
        # --- Entrenamiento ---
        model.train()
        train_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(Xb)
        train_loss /= len(Xt)

        # --- Validación ---
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv), yv).item()
            val_acc = (model(Xv).argmax(dim=1) == yv).float().mean().item()

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                "  MLP Época %3d/%d | loss=%.4f | val_loss=%.4f | val_acc=%.1f%%",
                epoch, epochs, train_loss, val_loss, val_acc * 100,
            )

        # Early stopping
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_acc = val_acc
            epochs_no_improve = 0
            history["best_epoch"] = epoch
            save_weights(model, output_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log.info(
                    "  MLP Early stopping en época %d (no mejora en %d épocas).",
                    epoch, patience,
                )
                break

    history["best_val_loss"] = best_val_loss
    history["best_val_acc"] = best_acc
    return history


# ---------------------------------------------------------------------------
# Entrenamiento CNN
# ---------------------------------------------------------------------------

def _train_cnn(
    X_tr, y_tr, X_va, y_va,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    patch_size: int,
    device: str | None,
    output_path: Path,
) -> dict:
    """Entrena la CNN multi-escala con early stopping y aumentación básica.

    Returns:
        Diccionario con historial de pérdidas y métricas de la mejor época.
    """
    import numpy as np

    torch = _import_torch()
    from roofscan.core.deteccion.dl.material_classifier import (
        N_CLASSES, N_BANDS,
        build_material_cnn, save_weights,
    )
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info("Entrenando CNN multi-escala en device=%s", dev)

    # --- Aumentación manual (flips horizontales/verticales en el batch de train) ---
    def _augment(Xb):
        """Flips aleatorios H/V sobre un batch de parches."""
        B = Xb.size(0)
        mask_h = torch.rand(B) < 0.5
        mask_v = torch.rand(B) < 0.5
        Xb[mask_h] = torch.flip(Xb[mask_h], dims=[3])   # flip horizontal
        Xb[mask_v] = torch.flip(Xb[mask_v], dims=[2])   # flip vertical
        return Xb

    # Datasets
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    Xv = torch.tensor(X_va, dtype=torch.float32).to(dev)
    yv = torch.tensor(y_va, dtype=torch.long).to(dev)

    sw = _sample_weights(y_tr, N_CLASSES)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sw, dtype=torch.double),
        num_samples=len(sw),
        replacement=True,
    )
    loader = DataLoader(
        TensorDataset(Xt, yt),
        batch_size=min(batch_size, len(Xt)),
        sampler=sampler,
    )

    cw = torch.tensor(_class_weights(y_tr, N_CLASSES), dtype=torch.float32).to(dev)
    criterion = torch.nn.CrossEntropyLoss(weight=cw)

    model = build_material_cnn(patch_size=patch_size, device=str(dev))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=patience // 3, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_acc = 0.0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "best_epoch": 0}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(dev), yb.to(dev)
            Xb = _augment(Xb)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(Xb)
        train_loss /= len(Xt)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv), yv).item()
            val_acc = (model(Xv).argmax(dim=1) == yv).float().mean().item()

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                "  CNN Época %3d/%d | loss=%.4f | val_loss=%.4f | val_acc=%.1f%%",
                epoch, epochs, train_loss, val_loss, val_acc * 100,
            )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_acc = val_acc
            epochs_no_improve = 0
            history["best_epoch"] = epoch
            save_weights(model, output_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log.info(
                    "  CNN Early stopping en época %d (no mejora en %d épocas).",
                    epoch, patience,
                )
                break

    history["best_val_loss"] = best_val_loss
    history["best_val_acc"] = best_acc
    return history


# ---------------------------------------------------------------------------
# Reporte de métricas
# ---------------------------------------------------------------------------

def _report_metrics(y_true, y_pred_logits, split_name: str) -> None:
    """Imprime precisión y matriz de confusión simplificada."""
    import numpy as np
    from roofscan.core.deteccion.dl.material_classifier import MATERIAL_LABELS, N_CLASSES

    torch = _import_torch()
    with torch.no_grad():
        y_pred = torch.tensor(y_pred_logits).argmax(dim=1).numpy()

    acc = (y_pred == y_true).mean() * 100
    log.info("  Precisión %s: %.1f%%", split_name, acc)

    # Precisión por clase
    for i, label in enumerate(MATERIAL_LABELS):
        mask = y_true == i
        if mask.sum() == 0:
            continue
        cls_acc = (y_pred[mask] == i).mean() * 100
        log.info("    %-28s %.1f%%  (n=%d)", label, cls_acc, mask.sum())


def _import_torch():
    """Importa PyTorch con mensaje claro si no está instalado."""
    try:
        import torch
        return torch
    except ImportError:
        log.error(
            "PyTorch no está instalado.\n"
            "  Instalalo con: pip install torch torchvision\n"
            "  O con GPU NVIDIA: pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu121"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if args.no_mlp and args.no_cnn:
        log.error("Se especificaron --no-mlp y --no-cnn. Al menos uno debe entrenarse.")
        sys.exit(1)

    # ── Carga de datos ──────────────────────────────────────────────────────
    image_data = _load_image(args.image)
    image_crs = str(image_data.get("crs", "EPSG:32720"))
    gdf = _load_labels(args.labels, args.label_col, image_crs)
    gdf = _validate_and_filter_labels(gdf, args.label_col)

    if len(gdf) < MIN_SAMPLES_TOTAL:
        log.error(
            "Se necesitan al menos %d muestras etiquetadas para entrenar. "
            "Encontradas: %d.",
            MIN_SAMPLES_TOTAL, len(gdf),
        )
        sys.exit(1)

    # ── Extracción de features ──────────────────────────────────────────────
    X_mlp, y_mlp, X_cnn, y_cnn = _extract_features(
        gdf, image_data, args.label_col, args.patch_size,
    )

    # ── Verificar disponibilidad de PyTorch antes de entrenar ───────────────
    _import_torch()

    # ── Entrenamiento MLP ───────────────────────────────────────────────────
    if not args.no_mlp:
        log.info("=" * 60)
        log.info("ENTRENAMIENTO MLP ESPECTRAL")
        log.info("  Muestras totales: %d", len(X_mlp))

        if len(X_mlp) < MIN_SAMPLES_TOTAL:
            log.warning(
                "Solo %d muestras MLP (mínimo recomendado: %d). "
                "Los resultados pueden ser poco confiables.",
                len(X_mlp), MIN_SAMPLES_TOTAL,
            )

        X_tr_m, y_tr_m, X_va_m, y_va_m = _train_val_split(
            X_mlp, y_mlp, args.val_split,
        )
        log.info(
            "  Train: %d | Val: %d | LR: %.0e | Épocas: %d",
            len(X_tr_m), len(X_va_m), args.lr, args.epochs,
        )
        args.output_mlp.parent.mkdir(parents=True, exist_ok=True)

        hist_mlp = _train_mlp(
            X_tr_m, y_tr_m, X_va_m, y_va_m,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            patience=args.patience,
            device=args.device,
            output_path=args.output_mlp,
        )

        log.info("=" * 60)
        log.info("MLP — Resumen:")
        log.info("  Mejor época:       %d", hist_mlp["best_epoch"])
        log.info("  Mejor val_loss:    %.4f", hist_mlp["best_val_loss"])
        log.info("  Mejor val_acc:     %.1f%%", hist_mlp["best_val_acc"] * 100)
        if args.output_mlp.exists():
            log.info(
                "  Pesos guardados:   %s (%.1f KB)",
                args.output_mlp.resolve(),
                args.output_mlp.stat().st_size / 1e3,
            )

    # ── Entrenamiento CNN ───────────────────────────────────────────────────
    if not args.no_cnn:
        log.info("=" * 60)
        log.info("ENTRENAMIENTO CNN MULTI-ESCALA")
        log.info("  Muestras totales: %d", len(X_cnn))

        if len(X_cnn) < MIN_SAMPLES_CNN:
            log.warning(
                "Solo %d muestras CNN válidas (mínimo recomendado: %d).\n"
                "  El MLP espectral es preferible con pocos datos.\n"
                "  Considerá entrenar solo el MLP con --no-cnn.",
                len(X_cnn), MIN_SAMPLES_CNN,
            )

        if len(X_cnn) == 0:
            log.error(
                "No hay parches CNN válidos. Verificá que los polígonos etiquetados "
                "estén dentro de la imagen."
            )
        else:
            X_tr_c, y_tr_c, X_va_c, y_va_c = _train_val_split(
                X_cnn, y_cnn, args.val_split,
            )
            log.info(
                "  Train: %d | Val: %d | LR: %.0e | Épocas: %d | Parche: %dpx",
                len(X_tr_c), len(X_va_c), args.lr, args.epochs, args.patch_size,
            )
            args.output_cnn.parent.mkdir(parents=True, exist_ok=True)

            hist_cnn = _train_cnn(
                X_tr_c, y_tr_c, X_va_c, y_va_c,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                patience=args.patience,
                patch_size=args.patch_size,
                device=args.device,
                output_path=args.output_cnn,
            )

            log.info("=" * 60)
            log.info("CNN — Resumen:")
            log.info("  Mejor época:       %d", hist_cnn["best_epoch"])
            log.info("  Mejor val_loss:    %.4f", hist_cnn["best_val_loss"])
            log.info("  Mejor val_acc:     %.1f%%", hist_cnn["best_val_acc"] * 100)
            if args.output_cnn.exists():
                log.info(
                    "  Pesos guardados:   %s (%.1f KB)",
                    args.output_cnn.resolve(),
                    args.output_cnn.stat().st_size / 1e3,
                )

    # ── Instrucciones finales ───────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Entrenamiento completado.")
    log.info("")
    log.info("Para usar los modelos en batch_mensura.py:")
    cmd_parts = [
        "  python scripts/batch_mensura.py",
        "      --image <imagen_stacked.tif>",
        "      --parcelas <parcelas.gpkg>",
        "      --output <salida.csv>",
        "      --material",
    ]
    if not args.no_mlp and args.output_mlp.exists():
        cmd_parts.append(f"      --material-mlp {args.output_mlp}")
    if not args.no_cnn and args.output_cnn.exists():
        cmd_parts.append(f"      --material-cnn {args.output_cnn}")
    log.info("\n".join(cmd_parts))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
