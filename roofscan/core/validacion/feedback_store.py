"""Almacenamiento de pares (imagen, máscara) para reentrenamiento.

Gestiona el dataset de feedback acumulado por el usuario. Cada par guardado
consiste en:

- Un array de imagen ``float32 (6, H, W)`` normalizado [0, 1]
- Una máscara ``uint8 (H, W)`` con valores 0 (fondo) o 1 (techo)

Los archivos se guardan en::

    feedback_dir/
        images/   ← {nombre}.npy
        masks/    ← {nombre}.npy   (mismo nombre)

El nombre de archivo incluye el timestamp y un sufijo aleatorio para evitar
colisiones al guardar múltiples pares en la misma sesión.

Uso típico::

    from roofscan.core.validacion.feedback_store import save_feedback_pair, count_feedback_pairs
    from roofscan.config import FEEDBACK_DIR

    name = save_feedback_pair(image_array, mask_array, FEEDBACK_DIR)
    print(f"Guardado: {name} | Total: {count_feedback_pairs(FEEDBACK_DIR)} pares")
"""

import logging
import random
import string
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def save_feedback_pair(
    image_array: np.ndarray,
    mask_array: np.ndarray,
    feedback_dir: Path | str,
) -> str:
    """Guarda un par (imagen, máscara) en el dataset de feedback.

    Args:
        image_array: Array float32 ``(bandas, H, W)`` normalizado [0, 1].
                     Normalmente salida de ``normalize_s2()``.
        mask_array: Array booleano o uint8 ``(H, W)``. ``True``/1 = techo.
        feedback_dir: Directorio raíz del dataset
                      (se crearán ``images/`` y ``masks/`` si no existen).

    Returns:
        Nombre del archivo guardado (sin extensión), igual para imagen y máscara.

    Raises:
        ValueError: Si los arrays no tienen formas compatibles.
    """
    feedback_dir = Path(feedback_dir)
    img_dir = feedback_dir / "images"
    msk_dir = feedback_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    _validate_pair(image_array, mask_array)

    name = _generate_name()

    # Imagen: asegurar float32
    img = image_array.astype(np.float32)

    # Máscara: asegurar uint8 binario (0/1)
    if mask_array.dtype == bool:
        msk = mask_array.astype(np.uint8)
    else:
        msk = (mask_array > 0).astype(np.uint8)

    np.save(img_dir / f"{name}.npy", img)
    np.save(msk_dir / f"{name}.npy", msk)

    n_total = count_feedback_pairs(feedback_dir)
    log.info(
        "Feedback guardado: %s | imagen=%s | techo=%.1f%% | total=%d pares",
        name,
        img.shape,
        100.0 * msk.sum() / msk.size,
        n_total,
    )
    return name


def list_feedback_pairs(feedback_dir: Path | str) -> list[str]:
    """Lista los nombres de todos los pares completos en el dataset.

    Solo incluye pares donde existe tanto imagen como máscara.

    Args:
        feedback_dir: Directorio raíz del dataset.

    Returns:
        Lista de nombres (sin extensión) ordenada alfabéticamente.
        Vacía si no hay pares o el directorio no existe.
    """
    feedback_dir = Path(feedback_dir)
    img_dir = feedback_dir / "images"
    msk_dir = feedback_dir / "masks"

    if not img_dir.is_dir() or not msk_dir.is_dir():
        return []

    img_names = {p.stem for p in img_dir.glob("*.npy")}
    msk_names = {p.stem for p in msk_dir.glob("*.npy")}
    return sorted(img_names & msk_names)


def count_feedback_pairs(feedback_dir: Path | str) -> int:
    """Devuelve el número de pares completos en el dataset.

    Args:
        feedback_dir: Directorio raíz del dataset.

    Returns:
        Número de pares imagen/máscara coincidentes.
    """
    return len(list_feedback_pairs(feedback_dir))


def delete_feedback_pair(name: str, feedback_dir: Path | str) -> bool:
    """Elimina un par imagen/máscara del dataset.

    Args:
        name: Nombre del par (sin extensión).
        feedback_dir: Directorio raíz del dataset.

    Returns:
        ``True`` si se eliminó al menos un archivo, ``False`` si no existía.
    """
    feedback_dir = Path(feedback_dir)
    deleted = False
    for subdir in ("images", "masks"):
        path = feedback_dir / subdir / f"{name}.npy"
        if path.exists():
            path.unlink()
            deleted = True
    if deleted:
        log.info("Feedback eliminado: %s", name)
    return deleted


def load_feedback_pair(
    name: str,
    feedback_dir: Path | str,
) -> tuple[np.ndarray, np.ndarray]:
    """Carga un par específico del dataset.

    Args:
        name: Nombre del par (sin extensión).
        feedback_dir: Directorio raíz del dataset.

    Returns:
        Tupla ``(image_array, mask_array)``.

    Raises:
        FileNotFoundError: Si alguno de los archivos no existe.
    """
    feedback_dir = Path(feedback_dir)
    img_path = feedback_dir / "images" / f"{name}.npy"
    msk_path = feedback_dir / "masks" / f"{name}.npy"

    if not img_path.exists():
        raise FileNotFoundError(f"No se encontró imagen de feedback: {img_path}")
    if not msk_path.exists():
        raise FileNotFoundError(f"No se encontró máscara de feedback: {msk_path}")

    return np.load(img_path), np.load(msk_path)


def feedback_stats(feedback_dir: Path | str) -> dict:
    """Calcula estadísticas del dataset de feedback acumulado.

    Args:
        feedback_dir: Directorio raíz del dataset.

    Returns:
        Dict con:

        - ``n_pairs``: número de pares.
        - ``total_pixels``: píxeles totales en todas las máscaras.
        - ``roof_pixels``: píxeles de techo en todas las máscaras.
        - ``roof_pct``: porcentaje de techo sobre el total.
    """
    pairs = list_feedback_pairs(feedback_dir)
    if not pairs:
        return {"n_pairs": 0, "total_pixels": 0, "roof_pixels": 0, "roof_pct": 0.0}

    feedback_dir = Path(feedback_dir)
    total_px = 0
    roof_px = 0
    for name in pairs:
        msk_path = feedback_dir / "masks" / f"{name}.npy"
        if msk_path.exists():
            msk = np.load(msk_path)
            total_px += msk.size
            roof_px += int((msk > 0).sum())

    roof_pct = 100.0 * roof_px / total_px if total_px > 0 else 0.0
    return {
        "n_pairs": len(pairs),
        "total_pixels": total_px,
        "roof_pixels": roof_px,
        "roof_pct": round(roof_pct, 2),
    }


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _generate_name() -> str:
    """Genera un nombre de archivo único basado en timestamp + sufijo aleatorio."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase, k=4))
    return f"{ts}_{suffix}"


def _validate_pair(image_array: np.ndarray, mask_array: np.ndarray) -> None:
    """Verifica que imagen y máscara sean compatibles."""
    if image_array.ndim != 3:
        raise ValueError(
            f"image_array debe ser 3D (bandas, H, W), recibido shape={image_array.shape}"
        )
    if mask_array.ndim != 2:
        raise ValueError(
            f"mask_array debe ser 2D (H, W), recibido shape={mask_array.shape}"
        )
    _, h_img, w_img = image_array.shape
    h_msk, w_msk = mask_array.shape
    if h_img != h_msk or w_img != w_msk:
        raise ValueError(
            f"Dimensiones espaciales incompatibles: imagen ({h_img}×{w_img}) "
            f"vs máscara ({h_msk}×{w_msk})."
        )
