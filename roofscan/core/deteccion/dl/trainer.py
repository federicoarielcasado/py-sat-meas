"""Entrenamiento y fine-tuning del modelo U-Net sobre datos locales.

Dataset esperado (``data/feedback/``):
    images/
        0001.npy   # float32 (6, H, W) normalizado [0, 1]
        0002.npy
        ...
    masks/
        0001.npy   # bool o uint8 (H, W), 1 = techo, 0 = fondo
        0002.npy
        ...

Los nombres de archivo deben coincidir entre ``images/`` y ``masks/``.

Función de pérdida:
    BCE + Dice combinados (Tversky α=0.5, β=0.5 = Dice), con pesos iguales.
    Esta combinación favorece tanto la precisión global (BCE) como la
    superposición espacial (Dice), especialmente útil con datos desbalanceados.

Uso típico::

    from roofscan.core.deteccion.dl.unet import build_unet, load_weights
    from roofscan.core.deteccion.dl.trainer import fine_tune
    from roofscan.config import FEEDBACK_DIR, MODELS_DIR

    model = build_unet(pretrained=True)
    fine_tune(model, dataset_dir=FEEDBACK_DIR, epochs=20)
"""

import logging
import random
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes de entrenamiento por defecto
# ---------------------------------------------------------------------------

DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE = 4
DEFAULT_VAL_SPLIT = 0.15      # 15 % del dataset para validación
DEFAULT_PATIENCE = 5          # early stopping: épocas sin mejora
SEED = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RoofDataset:
    """Dataset PyTorch de pares (imagen, máscara) guardados como .npy.

    La estructura esperada en ``dataset_dir`` es::

        dataset_dir/
            images/   ← arrays float32 (6, H, W) normalizados [0, 1]
            masks/    ← arrays uint8 o bool (H, W), 1=techo, 0=fondo

    Los archivos de imagen y máscara deben tener el **mismo nombre** (solo
    difieren en el directorio).

    Args:
        dataset_dir: Directorio raíz del dataset (contiene ``images/``
                     y ``masks/``).
        augment: Si ``True``, aplica flips y rotaciones aleatorias durante
                 el entrenamiento (aumentación de datos).
        indices: Lista opcional de índices para usar solo un subconjunto
                 (train/val split). Si es ``None`` usa todos los pares.

    Raises:
        FileNotFoundError: Si alguno de los subdirectorios no existe.
        ValueError: Si no se encuentran pares imagen/máscara.
    """

    def __init__(
        self,
        dataset_dir: Path | str,
        augment: bool = False,
        indices: list[int] | None = None,
    ):
        try:
            import torch  # noqa: F401  (verificación temprana)
        except ImportError as exc:
            raise ImportError("PyTorch no está instalado.") from exc

        dataset_dir = Path(dataset_dir)
        img_dir = dataset_dir / "images"
        msk_dir = dataset_dir / "masks"

        if not img_dir.is_dir():
            raise FileNotFoundError(
                f"No se encontró el directorio de imágenes: {img_dir}\n"
                "Creá la carpeta y añadí archivos .npy de imágenes."
            )
        if not msk_dir.is_dir():
            raise FileNotFoundError(
                f"No se encontró el directorio de máscaras: {msk_dir}\n"
                "Creá la carpeta y añadí archivos .npy de máscaras."
            )

        # Intersección de nombres de archivo (solo pares completos)
        img_names = {p.name for p in img_dir.glob("*.npy")}
        msk_names = {p.name for p in msk_dir.glob("*.npy")}
        common = sorted(img_names & msk_names)

        if not common:
            raise ValueError(
                f"No se encontraron pares imagen/máscara en {dataset_dir}.\n"
                "Asegurate de que los archivos .npy tengan el mismo nombre\n"
                "en las carpetas images/ y masks/."
            )

        self._img_dir = img_dir
        self._msk_dir = msk_dir
        self._files = common
        self.augment = augment

        # Aplicar índices de subconjunto si se proveen
        if indices is not None:
            self._files = [self._files[i] for i in indices]

        log.debug("RoofDataset: %d muestras | augment=%s", len(self._files), augment)

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int):
        """Devuelve un par ``(image_tensor, mask_tensor)``.

        Returns:
            image_tensor: float32 ``(6, H, W)``
            mask_tensor: float32 ``(1, H, W)`` con valores 0.0 o 1.0
        """
        import torch

        name = self._files[idx]
        img = np.load(self._img_dir / name).astype(np.float32)
        msk = np.load(self._msk_dir / name).astype(np.float32)

        # Normalizar máscara a [0, 1] por si viene como uint8 (0/255)
        if msk.max() > 1.0:
            msk = (msk > 127).astype(np.float32)

        # Asegurar shapes correctos
        if img.ndim == 2:
            img = img[np.newaxis]          # (1, H, W)
        if msk.ndim == 3:
            msk = msk[0]                   # (H, W)

        # Aumentación geométrica (solo en train)
        if self.augment:
            img, msk = _augment(img, msk)

        img_t = torch.from_numpy(img)
        msk_t = torch.from_numpy(msk).unsqueeze(0)   # (1, H, W)
        return img_t, msk_t


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def fine_tune(
    model,
    dataset_dir: Path | str,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_split: float = DEFAULT_VAL_SPLIT,
    device: str | None = None,
    save_path: Path | str | None = None,
    patience: int = DEFAULT_PATIENCE,
) -> dict:
    """Entrena (fine-tune) el modelo U-Net con el dataset local.

    Divide el dataset en entrenamiento y validación, optimiza con Adam y
    la pérdida BCE+Dice, y guarda el mejor modelo (por val_loss).

    Args:
        model: Modelo U-Net (output de :func:`~roofscan.core.deteccion.dl.unet.build_unet`).
        dataset_dir: Directorio raíz del dataset (``images/`` + ``masks/``).
        epochs: Número máximo de épocas de entrenamiento.
        lr: Tasa de aprendizaje para Adam.
        batch_size: Tamaño de lote.
        val_split: Fracción del dataset reservada para validación [0, 1).
        device: Dispositivo PyTorch. Si es ``None`` se detecta automáticamente.
        save_path: Ruta donde guardar el mejor modelo (``.pth``).
                   Si es ``None`` usa ``data/models/unet_best.pth``.
        patience: Épocas sin mejora en val_loss antes de detener
                  el entrenamiento (early stopping).

    Returns:
        Dict con historial de entrenamiento::

            {
                "train_loss": [float, ...],
                "val_loss":   [float, ...],
                "best_epoch": int,
                "best_val_loss": float,
            }

    Raises:
        FileNotFoundError: Si ``dataset_dir`` o sus subdirectorios no existen.
        ValueError: Si el dataset está vacío o la configuración es inválida.
        ImportError: Si PyTorch no está instalado.
    """
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "PyTorch no está instalado. Ejecutá:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc

    from roofscan.config import MODELS_DIR

    dataset_dir = Path(dataset_dir)
    save_path = Path(save_path) if save_path else MODELS_DIR / "unet_best.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ #
    # Split train / val
    # ------------------------------------------------------------------ #
    full_ds = RoofDataset(dataset_dir, augment=False)
    n = len(full_ds)
    if n < 2:
        raise ValueError(
            f"El dataset tiene solo {n} muestra(s). Se necesitan al menos 2 para\n"
            "poder separar entrenamiento y validación."
        )

    rng = random.Random(SEED)
    indices = list(range(n))
    rng.shuffle(indices)

    n_val = max(1, int(n * val_split))
    n_train = n - n_val
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_ds = RoofDataset(dataset_dir, augment=True, indices=train_idx)
    val_ds = RoofDataset(dataset_dir, augment=False, indices=val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(dev == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=(dev == "cuda"))

    log.info(
        "Fine-tune | muestras: total=%d train=%d val=%d | épocas=%d | lr=%.0e | device=%s",
        n, n_train, n_val, epochs, lr, dev,
    )

    # ------------------------------------------------------------------ #
    # Optimizador y scheduler
    # ------------------------------------------------------------------ #
    model = model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Reducir LR a la mitad si val_loss no mejora en patience//2 épocas
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, patience // 2),
    )

    # ------------------------------------------------------------------ #
    # Loop de entrenamiento
    # ------------------------------------------------------------------ #
    history: dict = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
    }
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        train_loss = _run_epoch(model, train_loader, optimizer, dev, training=True)

        # -- Val --
        model.eval()
        with torch.no_grad():
            val_loss = _run_epoch(model, val_loader, optimizer, dev, training=False)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        log.info(
            "Época %d/%d | train_loss=%.4f | val_loss=%.4f",
            epoch, epochs, train_loss, val_loss,
        )

        # Guardar mejor modelo
        if val_loss < history["best_val_loss"]:
            history["best_val_loss"] = val_loss
            history["best_epoch"] = epoch
            torch.save(model.state_dict(), save_path)
            log.info("  → Mejor modelo guardado (val_loss=%.4f) en %s", val_loss, save_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(
                    "Early stopping en época %d (sin mejora en %d épocas).",
                    epoch, patience,
                )
                break

    log.info(
        "Fine-tune completo | mejor época=%d | mejor val_loss=%.4f | modelo=%s",
        history["best_epoch"], history["best_val_loss"], save_path,
    )
    return history


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, optimizer, device: str, training: bool) -> float:
    """Ejecuta una época de entrenamiento o validación.

    Returns:
        Pérdida media sobre todos los batches.
    """
    import torch

    total_loss = 0.0
    n_batches = 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)               # (B, 1, H, W)
        loss = _bce_dice_loss(logits, masks)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def _bce_dice_loss(logits, targets, smooth: float = 1.0) -> "torch.Tensor":
    """Combina BCE con Dice loss (peso igual 0.5 + 0.5).

    Args:
        logits: Salida cruda del modelo ``(B, 1, H, W)``.
        targets: Máscaras objetivo float32 ``(B, 1, H, W)`` con valores 0/1.
        smooth: Factor de suavizado para evitar división por cero en Dice.

    Returns:
        Escalar tensor con la pérdida combinada.
    """
    import torch
    import torch.nn.functional as F

    bce = F.binary_cross_entropy_with_logits(logits, targets)

    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
    dice = dice.mean()

    return 0.5 * bce + 0.5 * dice


def _augment(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aplica aumentación geométrica aleatoria (flips y rotaciones de 90°).

    Args:
        img: Array float32 ``(C, H, W)``.
        mask: Array float32 ``(H, W)``.

    Returns:
        Tupla ``(img, mask)`` aumentados.
    """
    # Flip horizontal
    if random.random() < 0.5:
        img = img[:, :, ::-1].copy()
        mask = mask[:, ::-1].copy()

    # Flip vertical
    if random.random() < 0.5:
        img = img[:, ::-1, :].copy()
        mask = mask[::-1, :].copy()

    # Rotación 90°/180°/270°
    k = random.randint(0, 3)
    if k > 0:
        img = np.rot90(img, k=k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k=k, axes=(0, 1)).copy()

    return img, mask
