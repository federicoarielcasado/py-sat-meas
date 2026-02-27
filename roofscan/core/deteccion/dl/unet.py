"""Definición y gestión del modelo U-Net para segmentación de techos.

Usa segmentation-models-pytorch (smp) con encoder ResNet34 preentrenado
en ImageNet. El modelo acepta 6 bandas Sentinel-2 como entrada y produce
una máscara binaria (techo / no techo) por píxel.

Estrategia de preentrenamiento:
    - smp adapta automáticamente la primera capa convolucional cuando
      in_channels ≠ 3: promedia los pesos ImageNet existentes e inicializa
      los canales adicionales de forma compatible.
    - Para fine-tuning local se proveen save_weights() / load_weights().

Uso típico::

    from roofscan.core.deteccion.dl.unet import build_unet, load_weights
    from roofscan.config import MODELS_DIR

    model = build_unet(pretrained=True)
    model = load_weights(model, MODELS_DIR / "unet_best.pth")
    model.eval()
"""

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Número de bandas Sentinel-2 que usa el modelo
S2_IN_CHANNELS = 6

# Encoder por defecto: equilibrio rendimiento / velocidad en CPU
DEFAULT_ENCODER = "resnet34"

# Pesos del encoder (ImageNet). Primer uso descarga ~90 MB automáticamente.
DEFAULT_ENCODER_WEIGHTS = "imagenet"


def build_unet(
    in_channels: int = S2_IN_CHANNELS,
    encoder_name: str = DEFAULT_ENCODER,
    pretrained: bool = True,
    device: str | None = None,
) -> Any:
    """Construye el modelo U-Net con encoder ResNet34.

    Args:
        in_channels: Número de canales de entrada. Por defecto 6
                     (bandas B02-B03-B04-B08-B11-B12 de Sentinel-2).
        encoder_name: Nombre del encoder de smp (ej. ``"resnet34"``,
                      ``"efficientnet-b0"``).
        pretrained: Si ``True``, carga pesos ImageNet para el encoder.
                    Si ``False``, inicialización aleatoria (más rápido,
                    requiere más datos para converger).
        device: Dispositivo PyTorch (``"cpu"``, ``"cuda"``). Si es ``None``
                se detecta automáticamente.

    Returns:
        Modelo PyTorch :class:`torch.nn.Module` listo para entrenamiento
        o inferencia.

    Raises:
        ImportError: Si PyTorch o segmentation-models-pytorch no están instalados.
    """
    torch, smp = _import_deps()
    dev = _resolve_device(device)

    encoder_weights = DEFAULT_ENCODER_WEIGHTS if pretrained else None
    log.info(
        "Construyendo U-Net | encoder=%s | pesos=%s | in_channels=%d | device=%s",
        encoder_name, encoder_weights or "aleatorio", in_channels, dev,
    )

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=1,          # salida binaria (1 logit por píxel)
        activation=None,    # usamos BCEWithLogitsLoss → sigmoid en inferencia
    )

    model = model.to(dev)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info("Modelo construido | %.1f M parámetros | device=%s", n_params, dev)
    return model


def save_weights(model: Any, filepath: Path | str) -> None:
    """Guarda los pesos del modelo en disco.

    Args:
        model: Modelo PyTorch entrenado.
        filepath: Ruta del archivo ``.pth`` de destino.
    """
    torch, _ = _import_deps()
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)
    size_mb = filepath.stat().st_size / 1e6
    log.info("Pesos guardados: %s (%.1f MB)", filepath, size_mb)


def load_weights(
    model: Any,
    filepath: Path | str,
    device: str | None = None,
    strict: bool = True,
) -> Any:
    """Carga pesos guardados en un modelo existente.

    Args:
        model: Modelo PyTorch (misma arquitectura con la que se guardaron los pesos).
        filepath: Ruta al archivo ``.pth``.
        device: Dispositivo destino. Si es ``None`` se detecta automáticamente.
        strict: Si ``True`` (por defecto), falla si las claves no coinciden exactamente.

    Returns:
        El mismo modelo con los pesos cargados.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        RuntimeError: Si los pesos son incompatibles con la arquitectura.
    """
    torch, _ = _import_deps()
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"No se encontraron pesos del modelo en: {filepath}\n"
            "Para entrenar el modelo por primera vez usá fine_tune() del módulo trainer."
        )

    dev = _resolve_device(device)
    state_dict = torch.load(filepath, map_location=dev, weights_only=True)

    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Los pesos en {filepath} son incompatibles con la arquitectura actual.\n"
            f"Detalle: {exc}"
        ) from exc

    model = model.to(dev)
    log.info("Pesos cargados desde %s | device=%s", filepath, dev)
    return model


def get_device() -> str:
    """Devuelve el dispositivo disponible (``'cuda'`` o ``'cpu'``)."""
    torch, _ = _import_deps()
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _import_deps() -> tuple:
    """Importa torch y smp con mensajes de error descriptivos."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch no está instalado. Instalá según tu hardware:\n"
            "  CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  GPU: pip install torch\n"
            "Más info: https://pytorch.org/get-started/locally/"
        ) from exc
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "segmentation-models-pytorch no está instalado.\n"
            "Ejecutá: pip install segmentation-models-pytorch"
        ) from exc
    return torch, smp


def _resolve_device(device: str | None) -> str:
    """Resuelve el dispositivo: usa el argumento o detecta automáticamente."""
    if device is not None:
        return device
    return get_device()
