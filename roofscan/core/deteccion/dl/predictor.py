"""Inferencia del modelo U-Net sobre imágenes satelitales.

Estrategia de tiling:
    Las imágenes Sentinel-2 de una escena completa pueden ser grandes
    (hasta 10980×10980 px). Para evitar problemas de memoria, el predictor
    divide la imagen en parches solapados (tiles), predice cada uno y
    los une con Gaussian blending para eliminar artefactos en los bordes.

    Para imágenes pequeñas (≤ tile_size en ambas dimensiones), se predice
    directamente sin tiling.

Uso típico::

    from roofscan.core.deteccion.dl.unet import build_unet, load_weights
    from roofscan.core.deteccion.dl.predictor import predict_mask
    from roofscan.config import MODELS_DIR

    model = build_unet(pretrained=False)
    model = load_weights(model, MODELS_DIR / "unet_best.pth")
    mask = predict_mask(model, normalized_array, threshold=0.5)
"""

import logging

import numpy as np

log = logging.getLogger(__name__)

# Tamaño de tile por defecto en píxeles (poder de 2, compatible con U-Net)
DEFAULT_TILE_SIZE = 256
DEFAULT_OVERLAP = 32       # píxeles de solapamiento entre tiles
DEFAULT_THRESHOLD = 0.5    # umbral de probabilidad para binarizar


def predict_mask(
    model,
    array: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    device: str | None = None,
    batch_size: int = 4,
) -> np.ndarray:
    """Predice la máscara de techos con el modelo U-Net.

    Args:
        model: Modelo PyTorch U-Net (output de :func:`~roofscan.core.deteccion.dl.unet.build_unet`).
               Debe estar en modo ``eval()``.
        array: Array float32 normalizado ``(bandas, H, W)`` en rango [0, 1].
               Salida del pipeline de preprocesamiento.
        threshold: Umbral de probabilidad para binarizar la salida [0, 1].
                   Valores altos → menos detecciones (mayor precisión).
                   Valores bajos → más detecciones (mayor recall).
        tile_size: Tamaño en píxeles de cada tile cuadrado.
        overlap: Píxeles de solapamiento entre tiles adyacentes.
        device: Dispositivo PyTorch. Si es ``None`` se detecta automáticamente.
        batch_size: Número de tiles a procesar en paralelo (reduce el tiempo
                    en CPU y GPU).

    Returns:
        Array booleano 2D ``(H, W)`` con ``True`` donde se predice techo.

    Raises:
        ValueError: Si ``array`` no tiene el formato esperado.
        ImportError: Si PyTorch no está instalado.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch no está instalado. Ejecutá:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc

    _validate_array(array)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, H, W = array.shape

    model.eval()
    model = model.to(dev)

    log.info(
        "Predicción U-Net | shape=(%d,%d) | tile=%d | overlap=%d | device=%s",
        H, W, tile_size, overlap, dev,
    )

    with torch.no_grad():
        if H <= tile_size and W <= tile_size:
            prob_map = _predict_single(model, array, dev)
        else:
            prob_map = _predict_tiled(model, array, tile_size, overlap, dev, batch_size)

    mask = prob_map >= threshold
    n_detected = mask.sum()
    log.info(
        "Predicción completa | px detectados: %d (%.1f%%) | umbral=%.2f",
        n_detected, 100.0 * n_detected / mask.size, threshold,
    )
    return mask


def predict_proba(
    model,
    array: np.ndarray,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    device: str | None = None,
) -> np.ndarray:
    """Devuelve el mapa de probabilidades (sin binarizar).

    Útil para ajustar el umbral de decisión post-predicción.

    Args:
        model: Modelo U-Net en modo eval.
        array: Array float32 normalizado ``(bandas, H, W)``.
        tile_size: Tamaño de tile en píxeles.
        overlap: Solapamiento entre tiles.
        device: Dispositivo PyTorch.

    Returns:
        Array float32 2D ``(H, W)`` con valores en [0, 1].
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch no instalado.") from exc

    _validate_array(array)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, H, W = array.shape

    model.eval()
    model = model.to(dev)

    with torch.no_grad():
        if H <= tile_size and W <= tile_size:
            return _predict_single(model, array, dev)
        return _predict_tiled(model, array, tile_size, overlap, dev, batch_size=4)


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _predict_single(model, array: np.ndarray, device: str) -> np.ndarray:
    """Predice un array sin tiling (cabe entero en memoria)."""
    import torch

    tensor = _array_to_tensor(array, device)          # (1, C, H, W)
    logits = model(tensor)                             # (1, 1, H, W)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H, W)
    return probs.astype(np.float32)


def _predict_tiled(
    model,
    array: np.ndarray,
    tile_size: int,
    overlap: int,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Predice con ventana deslizante y Gaussian blending."""
    import torch

    _, H, W = array.shape
    step = tile_size - overlap

    # Acumuladores para blending
    prob_acc = np.zeros((H, W), dtype=np.float64)
    weight_acc = np.zeros((H, W), dtype=np.float64)
    gauss_kernel = _gaussian_kernel(tile_size)

    # Generar todos los tiles
    tiles = []
    positions = []
    for row_start in range(0, H, step):
        for col_start in range(0, W, step):
            row_end = min(row_start + tile_size, H)
            col_end = min(col_start + tile_size, W)
            row_start_c = max(0, row_end - tile_size)
            col_start_c = max(0, col_end - tile_size)

            patch = array[:, row_start_c:row_end, col_start_c:col_end]
            # Pad si el patch es más pequeño que tile_size
            patch = _pad_to_size(patch, tile_size)
            tiles.append(patch)
            positions.append((row_start_c, row_end, col_start_c, col_end))

    # Predecir en batches
    for i in range(0, len(tiles), batch_size):
        batch_patches = tiles[i:i + batch_size]
        batch_positions = positions[i:i + batch_size]

        batch_tensor = torch.stack(
            [torch.from_numpy(p).float() for p in batch_patches]
        ).to(device)

        logits = model(batch_tensor)
        probs_batch = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # (B, H, W)

        for j, (r0, r1, c0, c1) in enumerate(batch_positions):
            tile_h = r1 - r0
            tile_w = c1 - c0
            pred = probs_batch[j][:tile_h, :tile_w]
            weight = gauss_kernel[:tile_h, :tile_w]
            prob_acc[r0:r1, c0:c1] += pred * weight
            weight_acc[r0:r1, c0:c1] += weight

        log.debug("Tiling: procesados %d/%d tiles", min(i + batch_size, len(tiles)), len(tiles))

    # Normalizar por peso acumulado
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(weight_acc > 0, prob_acc / weight_acc, 0.0)

    return result.astype(np.float32)


def _array_to_tensor(array: np.ndarray, device: str):
    """Convierte un numpy array (C, H, W) a tensor (1, C, H, W)."""
    import torch
    return torch.from_numpy(array).float().unsqueeze(0).to(device)


def _pad_to_size(array: np.ndarray, size: int) -> np.ndarray:
    """Rellena un array (C, H, W) hasta (C, size, size) con ceros."""
    C, H, W = array.shape
    if H == size and W == size:
        return array
    padded = np.zeros((C, size, size), dtype=array.dtype)
    padded[:, :H, :W] = array
    return padded


def _gaussian_kernel(size: int, sigma_ratio: float = 0.25) -> np.ndarray:
    """Genera un kernel gaussiano 2D para blending de tiles."""
    sigma = size * sigma_ratio
    center = size / 2.0
    y, x = np.ogrid[:size, :size]
    kernel = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
    return kernel.astype(np.float32)


def _validate_array(array: np.ndarray) -> None:
    """Valida el array de entrada."""
    if array.ndim != 3:
        raise ValueError(
            f"El array debe ser 3D (bandas, H, W), recibido shape={array.shape}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(
            f"El array debe ser float (float32/float64), recibido {array.dtype}. "
            "Aplicá normalize_s2() antes de predecir."
        )
