"""Clasificación de material de techo usando firma espectral Sentinel-2.

Pipeline de dos estrategias complementarias:

1. **Espectral-MLP** (rápida): Extrae estadísticas espectrales de todos los
   píxeles dentro del polígono de techo (media, std, p25, p75 por banda + 5
   índices derivados = 29 features) y los pasa por un MLP liviano.
   Funciona para cualquier tamaño de techo, incluso de 1-2 píxeles.

2. **CNN multi-escala** (precisa): Extrae un parche espacial de 32×32 px
   centrado en el techo y aplica una CNN con ramas paralelas de filtros 3×3 y
   5×5 concatenadas (inspirada en Kim et al., 2021, MDPI Electronics).
   Más precisa para techos ≥ 50 m² donde hay contexto espacial.

Si ambos modelos están disponibles, las probabilidades se promedian.
Si ninguno está entrenado, se aplica una clasificación por reglas espectrales
calibradas para materiales comunes en Luján, Buenos Aires (fallback sin ML).

Materiales clasificados:
    - ``zinc_corrugado``:         chapa/zinc ondulado — alta reflectancia SWIR
    - ``losa_hormigon``:          losa plana de hormigón — reflectancia plana moderada
    - ``tejas_ceramica``:         tejas de arcilla cocida — perfil espectral rojizo
    - ``construccion_incompleta``: obra sin techo terminado o precaria

Uso típico::

    from roofscan.core.deteccion.dl.material_classifier import (
        build_material_cnn,
        build_material_mlp,
        classify_roof_materials,
        load_weights,
    )
    from roofscan.config import MODELS_DIR

    # Sin modelos entrenados (fallback espectral, siempre disponible)
    gdf = classify_roof_materials(image_data, gdf_roofs)

    # Con MLP entrenado
    mlp = build_material_mlp()
    mlp = load_weights(mlp, MODELS_DIR / "material_mlp.pth")
    mlp.eval()
    gdf = classify_roof_materials(image_data, gdf_roofs, mlp_model=mlp)

    # Con CNN entrenada
    cnn = build_material_cnn()
    cnn = load_weights(cnn, MODELS_DIR / "material_cnn.pth")
    cnn.eval()
    gdf = classify_roof_materials(image_data, gdf_roofs, cnn_model=cnn)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes: materiales y etiquetas
# ---------------------------------------------------------------------------

#: Clases de material en orden canónico (índice == clase del modelo).
MATERIAL_LABELS: list[str] = [
    "zinc_corrugado",           # 0 — alta reflectancia SWIR, B11 y B12 altos
    "losa_hormigon",            # 1 — reflectancia plana moderada, baja varianza
    "tejas_ceramica",           # 2 — perfil rojizo, B04/B03 elevado
    "construccion_incompleta",  # 3 — NDVI alto o reflectancias muy bajas
]

#: Mapeo etiqueta → índice (inverso de MATERIAL_LABELS).
MATERIAL_IDX: dict[str, int] = {lbl: i for i, lbl in enumerate(MATERIAL_LABELS)}

#: Número de clases de material.
N_CLASSES = len(MATERIAL_LABELS)

#: Nombre del archivo de pesos del MLP.
MLP_WEIGHTS_FILENAME = "material_mlp.pth"

#: Nombre del archivo de pesos de la CNN.
CNN_WEIGHTS_FILENAME = "material_cnn.pth"

#: Número de estadísticas por banda: media, std, p25, p75.
_N_STATS = 4

#: Número de bandas Sentinel-2 (B02, B03, B04, B08, B11, B12).
N_BANDS = 6

#: Número de índices espectrales derivados añadidos como features.
_N_INDICES = 5

#: Total de features de entrada para el MLP.
N_SPECTRAL_FEATURES = N_BANDS * _N_STATS + _N_INDICES  # 29

# Índices de posición de cada banda en el array [B02, B03, B04, B08, B11, B12]
_B02, _B03, _B04, _B08, _B11, _B12 = 0, 1, 2, 3, 4, 5


# ---------------------------------------------------------------------------
# Modelos PyTorch
# ---------------------------------------------------------------------------

def build_material_mlp(
    n_features: int = N_SPECTRAL_FEATURES,
    device: str | None = None,
) -> Any:
    """Construye el MLP clasificador de material espectral.

    Recibe 29 features espectrales (media/std/p25/p75 por banda + 5 índices)
    y clasifica el techo en 4 materiales. Recomendado como primera opción por
    su velocidad y por funcionar incluso en techos de 1–2 píxeles.

    Args:
        n_features: Número de features de entrada. Default: 29.
        device: Dispositivo PyTorch (``"cpu"``, ``"cuda"``). Si es ``None``,
                se detecta automáticamente.

    Returns:
        Modelo PyTorch ``nn.Module`` listo para entrenamiento o inferencia.

    Raises:
        ImportError: Si PyTorch no está instalado.
    """
    torch = _import_torch()
    nn = torch.nn
    dev = _resolve_device(device, torch)

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, N_CLASSES),
            )
            # Inicialización He (recomendada por Kim et al. 2021 para ReLU)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            return self.net(x)

    model = _MLP().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("MaterialMLP construido | %d parámetros | device=%s", n_params, dev)
    return model


def build_material_cnn(
    in_channels: int = N_BANDS,
    patch_size: int = 32,
    device: str | None = None,
) -> Any:
    """Construye la CNN multi-escala para clasificación de material de techo.

    Arquitectura inspirada en Kim et al. (2021, MDPI Electronics): ramas
    paralelas de filtros 3×3 y 5×5 concatenadas (módulo tipo Inception),
    seguidas de Global Average Pooling y capas FC. Adaptada para 6 bandas
    Sentinel-2 en parches de 32×32 px.

    Diseñada para parches de 32×32 px (= 320×320 m en Sentinel-2 10 m/px),
    centrados en el centroide del polígono de techo.

    Args:
        in_channels: Canales de entrada (bandas S2). Default: 6.
        patch_size: Tamaño del parche en píxeles. Default: 32.
        device: Dispositivo PyTorch. Si es ``None``, se detecta automáticamente.

    Returns:
        Modelo PyTorch ``nn.Module``.

    Raises:
        ImportError: Si PyTorch no está instalado.
    """
    torch = _import_torch()
    nn = torch.nn
    dev = _resolve_device(device, torch)

    class _MultiScaleBlock(nn.Module):
        """Extracción multi-escala con ramas 3×3 y 5×5 en paralelo."""

        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            half = out_ch // 2
            self.branch3 = nn.Sequential(
                nn.Conv2d(in_ch, half, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(half),
                nn.ReLU(inplace=True),
            )
            self.branch5 = nn.Sequential(
                nn.Conv2d(in_ch, half, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm2d(half),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return torch.cat([self.branch3(x), self.branch5(x)], dim=1)

    class _MaterialCNN(nn.Module):
        def __init__(self):
            super().__init__()
            # Capa inicial: resalta patrones de textura con MaxPool (paper §3)
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),        # 32×32 → 16×16
            )
            # Bloques multi-escala (3×3 || 5×5) con reducción espacial
            self.ms1 = _MultiScaleBlock(32, 64)     # → 64 ch, 16×16
            self.pool1 = nn.MaxPool2d(2)             # → 8×8
            self.ms2 = _MultiScaleBlock(64, 128)    # → 128 ch, 8×8
            self.pool2 = nn.MaxPool2d(2)             # → 4×4
            # Global Average Pooling → vector de 128
            self.gap = nn.AdaptiveAvgPool2d(1)
            # Cabeza clasificadora con Dropout 50% (paper §4.1: dropout=0.5)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(64, N_CLASSES),
            )
            # He initialization
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):
            x = self.stem(x)
            x = self.ms1(x)
            x = self.pool1(x)
            x = self.ms2(x)
            x = self.pool2(x)
            x = self.gap(x)
            return self.classifier(x)

    model = _MaterialCNN().to(dev)
    n_params = sum(p.numel() for p in model.parameters()) / 1e3
    log.info(
        "MaterialCNN construida | %.1f K parámetros | in_ch=%d | patch=%dpx | device=%s",
        n_params, in_channels, patch_size, dev,
    )
    return model


# ---------------------------------------------------------------------------
# Gestión de pesos
# ---------------------------------------------------------------------------

def save_weights(model: Any, filepath: Path | str) -> None:
    """Guarda los pesos del clasificador de material en disco.

    Args:
        model: Modelo PyTorch entrenado (MaterialMLP o MaterialCNN).
        filepath: Ruta del archivo ``.pth`` de destino.
    """
    torch = _import_torch()
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)
    size_kb = filepath.stat().st_size / 1e3
    log.info("Pesos de material classifier guardados: %s (%.1f KB)", filepath, size_kb)


def load_weights(
    model: Any,
    filepath: Path | str,
    device: str | None = None,
    strict: bool = True,
) -> Any:
    """Carga pesos guardados en un modelo de clasificador de material.

    Args:
        model: Instancia del modelo (misma arquitectura que al guardar).
        filepath: Ruta al archivo ``.pth``.
        device: Dispositivo destino. Si es ``None``, se detecta automáticamente.
        strict: Si ``True``, falla si las claves no coinciden exactamente.

    Returns:
        El modelo con los pesos cargados.

    Raises:
        FileNotFoundError: Si el archivo ``.pth`` no existe.
        RuntimeError: Si los pesos son incompatibles con la arquitectura.
    """
    torch = _import_torch()
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"No se encontraron pesos del clasificador de material en: {filepath}\n"
            "Entrenalo primero con fine_tune() o recolectá muestras con "
            "collect_training_samples()."
        )
    dev = _resolve_device(device, torch)
    state_dict = torch.load(filepath, map_location=dev, weights_only=True)
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Pesos incompatibles con la arquitectura actual.\nDetalle: {exc}"
        ) from exc
    model = model.to(dev)
    log.info("Pesos de material classifier cargados desde %s | device=%s", filepath, dev)
    return model


# ---------------------------------------------------------------------------
# Extracción de features
# ---------------------------------------------------------------------------

def extract_spectral_stats(
    array: np.ndarray,
    transform: Any,
    polygon: Any,
    nodata: float | None = None,
) -> np.ndarray | None:
    """Extrae estadísticas espectrales de los píxeles dentro de un polígono.

    Computa media, std, percentil 25 y percentil 75 para cada una de las
    6 bandas Sentinel-2, más 5 índices espectrales derivados de las medias.
    Total: 6×4 + 5 = 29 features.

    Índices incluidos: NDVI, NDBI, BSI (Bare Soil Index), ratio B11/B12,
    ratio B04/B03.

    Si el polígono no cubre ningún píxel completo, se usa el centroide como
    punto representativo.

    Args:
        array: Array float32 normalizado ``(6, H, W)``. Bandas en orden
               [B02, B03, B04, B08, B11, B12].
        transform: Transformada affine rasterio del array.
        polygon: Polígono Shapely en el mismo CRS que el array.
        nodata: Valor a excluir como no-dato. Si es ``None``, no se filtra.

    Returns:
        Vector float32 de 29 features, o ``None`` si no hay píxeles válidos.

    Raises:
        ImportError: Si rasterio no está instalado.
    """
    try:
        from rasterio.features import geometry_mask
        from rasterio.transform import rowcol
    except ImportError as exc:
        raise ImportError("rasterio es necesario para extract_spectral_stats.") from exc

    n_bands, H, W = array.shape

    # Máscara de píxeles dentro del polígono (True = dentro)
    try:
        pixel_mask = ~geometry_mask(
            [polygon],
            transform=transform,
            invert=False,
            out_shape=(H, W),
        )
    except Exception:
        return None

    if pixel_mask.sum() == 0:
        # El polígono no cubre ningún píxel completo (sub-pixel o fuera) →
        # intentar con el centroide sin clampear para detectar casos fuera de imagen
        try:
            row, col = rowcol(transform, polygon.centroid.x, polygon.centroid.y)
            row, col = int(row), int(col)
            if not (0 <= row < H and 0 <= col < W):
                return None
            pixel_mask[row, col] = True
        except Exception:
            return None

    pixels = array[:, pixel_mask]  # (6, n_pixels)

    if nodata is not None:
        valid = ~np.any(pixels == nodata, axis=0)
        pixels = pixels[:, valid]

    if pixels.shape[1] == 0:
        return None

    # Estadísticas por banda: 4 × 6 = 24 features
    means = pixels.mean(axis=1)
    stds  = pixels.std(axis=1)
    p25s  = np.percentile(pixels, 25, axis=1)
    p75s  = np.percentile(pixels, 75, axis=1)

    stats = np.concatenate([means, stds, p25s, p75s])  # (24,)

    # Índices espectrales sobre las medias de banda: 5 features
    m02, m03, m04, m08, m11, m12 = (means[i] for i in range(N_BANDS))

    ndvi          = _safe_ratio(m08 - m04, m08 + m04)
    ndbi          = _safe_ratio(m11 - m08, m11 + m08)
    bsi           = _safe_ratio((m11 + m04) - (m08 + m02), (m11 + m04) + (m08 + m02))
    ratio_b11_b12 = _safe_ratio(m11, m12)
    ratio_b04_b03 = _safe_ratio(m04, m03)

    indices = np.array([ndvi, ndbi, bsi, ratio_b11_b12, ratio_b04_b03], dtype=np.float32)
    return np.concatenate([stats, indices]).astype(np.float32)  # (29,)


def extract_roof_patch(
    array: np.ndarray,
    transform: Any,
    polygon: Any,
    patch_size: int = 32,
) -> np.ndarray | None:
    """Extrae un parche espacial de ``patch_size×patch_size`` px centrado en el techo.

    El parche se centra en el centroide del polígono. Si el centroide cae fuera
    de los límites de la imagen, retorna ``None``. Aplica zero-padding si el
    centro está cerca del borde.

    Args:
        array: Array float32 normalizado ``(6, H, W)``.
        transform: Transformada affine rasterio.
        polygon: Polígono Shapely en el mismo CRS que el array.
        patch_size: Tamaño del parche en píxeles. Default: 32.

    Returns:
        Array float32 ``(6, patch_size, patch_size)`` o ``None``.

    Raises:
        ImportError: Si rasterio no está instalado.
    """
    try:
        from rasterio.transform import rowcol
    except ImportError as exc:
        raise ImportError("rasterio es necesario para extract_roof_patch.") from exc

    n_bands, H, W = array.shape

    try:
        row_c, col_c = rowcol(transform, polygon.centroid.x, polygon.centroid.y)
        row_c, col_c = int(row_c), int(col_c)
    except Exception:
        return None

    if not (0 <= row_c < H and 0 <= col_c < W):
        return None

    half = patch_size // 2
    r0 = max(0, row_c - half)
    c0 = max(0, col_c - half)
    r1 = min(H, r0 + patch_size)
    c1 = min(W, c0 + patch_size)

    crop = array[:, r0:r1, c0:c1]

    # Pad si el recorte es más pequeño que patch_size (borde de imagen)
    if crop.shape[1] != patch_size or crop.shape[2] != patch_size:
        padded = np.zeros((n_bands, patch_size, patch_size), dtype=array.dtype)
        padded[:, : crop.shape[1], : crop.shape[2]] = crop
        return padded

    return crop


# ---------------------------------------------------------------------------
# API principal
# ---------------------------------------------------------------------------

def classify_roof_materials(
    image_data: dict,
    gdf_roofs: Any,
    mlp_model: Any | None = None,
    cnn_model: Any | None = None,
    device: str | None = None,
) -> Any:
    """Clasifica el material de cada techo detectado.

    Agrega las columnas ``material``, ``material_score`` y ``material_method``
    al GeoDataFrame de techos.

    Estrategia (en orden de prioridad):
        - Si hay CNN + MLP: usa ambos y promedia probabilidades.
        - Si solo CNN: usa CNN sobre parche 32×32.
        - Si solo MLP: usa MLP sobre estadísticas espectrales.
        - Sin modelos: clasificación por reglas espectrales (fallback inmediato,
          sin necesidad de PyTorch ni datos de entrenamiento).

    Args:
        image_data: Diccionario retornado por ``load_geotiff()`` o
                    ``run_preprocessing()``. Debe contener ``array``
                    (float32, 6 bandas normalizadas) y ``transform``.
        gdf_roofs: GeoDataFrame con polígonos de techos individuales en el
                   CRS de trabajo (EPSG:32720 o 32721).
        mlp_model: Modelo MaterialMLP opcional, ya en modo ``eval()``.
        cnn_model: Modelo MaterialCNN opcional, ya en modo ``eval()``.
        device: Dispositivo PyTorch. Si es ``None``, se detecta automáticamente.

    Returns:
        Copia del GeoDataFrame con columnas adicionales:
            - ``material``: etiqueta del material (``str``).
            - ``material_score``: confianza de la predicción [0, 1] (``float``).
            - ``material_method``: método usado — ``"cnn+mlp"``, ``"cnn"``,
              ``"mlp"`` o ``"spectral_rules"``.

    Raises:
        KeyError: Si ``image_data`` no contiene ``array`` o ``transform``.
        ValueError: Si ``array`` no tiene exactamente 6 bandas.
    """
    array = image_data["array"]
    transform = image_data["transform"]

    if array.ndim != 3 or array.shape[0] != N_BANDS:
        raise ValueError(
            f"array debe tener shape (6, H, W), recibido {array.shape}. "
            "Asegurate de usar normalize_s2() y de tener las 6 bandas S2."
        )

    has_dl = (mlp_model is not None or cnn_model is not None)
    if has_dl:
        try:
            _import_torch()
        except ImportError:
            log.warning("PyTorch no instalado — usando reglas espectrales.")
            has_dl = False
            mlp_model = None
            cnn_model = None

    result = gdf_roofs.copy()
    materials, scores, methods = [], [], []

    for _, row in result.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            materials.append("construccion_incompleta")
            scores.append(0.0)
            methods.append("spectral_rules")
            continue

        label, score, method = _classify_single(
            array, transform, geom,
            mlp_model=mlp_model if has_dl else None,
            cnn_model=cnn_model if has_dl else None,
            device=device,
        )
        materials.append(label)
        scores.append(score)
        methods.append(method)

    result["material"] = materials
    result["material_score"] = np.array(scores, dtype=np.float32)
    result["material_method"] = methods

    dist = result["material"].value_counts().to_dict()
    method_used = (
        "cnn+mlp" if (mlp_model and cnn_model) else
        "cnn"    if cnn_model else
        "mlp"    if mlp_model else
        "spectral_rules"
    )
    log.info(
        "classify_roof_materials | n=%d | método=%s | distribución: %s",
        len(result), method_used, dist,
    )
    return result


# ---------------------------------------------------------------------------
# Helpers privados: clasificación de un polígono individual
# ---------------------------------------------------------------------------

def _classify_single(
    array: np.ndarray,
    transform: Any,
    polygon: Any,
    mlp_model: Any | None,
    cnn_model: Any | None,
    device: str | None,
) -> tuple[str, float, str]:
    """Clasifica un único polígono. Devuelve ``(label, score, method)``."""
    probs_cnn: np.ndarray | None = None
    probs_mlp: np.ndarray | None = None

    if cnn_model is not None:
        patch = extract_roof_patch(array, transform, polygon, patch_size=32)
        if patch is not None:
            probs_cnn = _infer(cnn_model, patch[np.newaxis], device)

    if mlp_model is not None:
        feats = extract_spectral_stats(array, transform, polygon)
        if feats is not None:
            probs_mlp = _infer(mlp_model, feats[np.newaxis], device)

    if probs_cnn is not None and probs_mlp is not None:
        probs, method = (probs_cnn + probs_mlp) / 2.0, "cnn+mlp"
    elif probs_cnn is not None:
        probs, method = probs_cnn, "cnn"
    elif probs_mlp is not None:
        probs, method = probs_mlp, "mlp"
    else:
        feats = extract_spectral_stats(array, transform, polygon)
        if feats is None:
            return "construccion_incompleta", 0.0, "spectral_rules"
        probs, method = _spectral_rule_probs(feats), "spectral_rules"

    idx = int(np.argmax(probs))
    return MATERIAL_LABELS[idx], float(probs[idx]), method


def _infer(model: Any, x: np.ndarray, device: str | None) -> np.ndarray:
    """Inferencia del modelo: retorna probabilidades softmax ``(N_CLASSES,)``."""
    import torch
    import torch.nn.functional as F

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    tensor = torch.from_numpy(x).float().to(dev)
    with torch.no_grad():
        logits = model(tensor)               # (1, N_CLASSES)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs.astype(np.float32)


def _spectral_rule_probs(features: np.ndarray) -> np.ndarray:
    """Asigna probabilidades por reglas espectrales (fallback sin ML).

    Heurística calibrada para materiales comunes en Luján, Buenos Aires:

    - **Zinc corrugado**: NDBI alto (> 0.10), B11 elevado, ratio B11/B12 ≈ 1.
    - **Losa hormigón**: NDBI moderado (~0.05), baja varianza espectral, BSI positivo.
    - **Tejas cerámicas**: ratio B04/B03 > 1.10 (componente roja de la arcilla).
    - **Construcción incompleta**: NDVI alto (> 0.10) o reflectancias muy bajas.

    Args:
        features: Vector de 29 features (salida de ``extract_spectral_stats``).

    Returns:
        Array float32 de probabilidades ``(4,)`` que suma 1.
    """
    means = features[:N_BANDS]          # medias de las 6 bandas → posiciones 0–5
    # Índices espectrales: posiciones 24–28
    ndvi          = float(features[24])
    ndbi          = float(features[25])
    bsi           = float(features[26])
    ratio_b11_b12 = float(features[27])
    ratio_b04_b03 = float(features[28])
    mean_b11      = float(means[_B11])

    scores = np.zeros(N_CLASSES, dtype=np.float32)

    # zinc_corrugado: NDBI alto, B11 elevado, B11/B12 cercano a 1
    scores[MATERIAL_IDX["zinc_corrugado"]] = (
        max(0.0, ndbi) * 2.0
        + mean_b11 * 1.5
        + max(0.0, 1.0 - abs(ratio_b11_b12 - 1.0)) * 0.5
    )

    # losa_hormigon: BSI positivo, NDBI moderado bajo, baja varianza en B11
    std_b11 = float(features[N_BANDS + _B11])  # posición 6+4=10
    scores[MATERIAL_IDX["losa_hormigon"]] = (
        max(0.0, bsi) * 1.5
        + max(0.0, 0.1 - abs(ndbi - 0.05)) * 3.0
        + max(0.0, 0.1 - std_b11) * 1.0
    )

    # tejas_ceramica: ratio B04/B03 elevado (reflectancia rojiza de la arcilla)
    scores[MATERIAL_IDX["tejas_ceramica"]] = (
        max(0.0, ratio_b04_b03 - 1.0) * 2.0
        + max(0.0, -ndbi) * 0.5
    )

    # construccion_incompleta: NDVI alto o todo bajo
    scores[MATERIAL_IDX["construccion_incompleta"]] = (
        max(0.0, ndvi - 0.10) * 2.0
        + max(0.0, -bsi) * 0.5
    )

    # Softmax para convertir a probabilidades
    scores = np.exp(scores - scores.max())
    total = scores.sum()
    if total > 0:
        scores /= total
    else:
        scores[:] = 1.0 / N_CLASSES

    return scores


# ---------------------------------------------------------------------------
# Helpers de PyTorch y aritmética segura
# ---------------------------------------------------------------------------

def _import_torch() -> Any:
    """Importa torch con mensaje de error descriptivo."""
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch no está instalado. Instalá según tu hardware:\n"
            "  CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  GPU: pip install torch\n"
            "Más info: https://pytorch.org/get-started/locally/"
        ) from exc


def _resolve_device(device: str | None, torch: Any) -> str:
    """Resuelve el dispositivo: usa el argumento o detecta automáticamente."""
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _safe_ratio(num: float, den: float, eps: float = 1e-8) -> float:
    """División segura: retorna 0 si el denominador es ~0."""
    return float(num / (den + eps))
