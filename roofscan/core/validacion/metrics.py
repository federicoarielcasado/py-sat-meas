"""Métricas de validación: comparación automático vs manual.

Calcula métricas estándar de segmentación (IoU, Precision, Recall, F1)
y de área (MAPE) para cuantificar la precisión del motor de detección.

Los resultados se acumulan en un log CSV en ``data/feedback/validation_log.csv``
para seguimiento histórico de la precisión del sistema.

Uso típico::

    from roofscan.core.validacion.metrics import validate, log_validation_result
    from roofscan.config import FEEDBACK_DIR

    result = validate(
        pred_area_m2=450.0,
        manual_area_m2=420.0,
        pred_mask=bool_mask,
        label="Calle San Martín 123",
    )
    log_validation_result(result, FEEDBACK_DIR / "validation_log.csv")
    print(f"Error de área: {result.mape_pct:.1f}%")
"""

import csv
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Columnas del log CSV (orden canónico)
_LOG_FIELDS = [
    "timestamp", "label",
    "pred_area_m2", "manual_area_m2", "mape_pct",
    "iou", "precision", "recall", "f1",
]


@dataclass
class ValidationResult:
    """Resultado de una sesión de validación.

    Attributes:
        timestamp: Marca de tiempo ISO 8601.
        pred_area_m2: Área total detectada automáticamente en m².
        manual_area_m2: Área medida manualmente por el usuario en m².
        mape_pct: Error porcentual absoluto medio (%).
        iou: Intersection over Union píxel a píxel (None si no hay máscara GT).
        precision: Precisión píxel a píxel (None si no hay máscara GT).
        recall: Recall píxel a píxel (None si no hay máscara GT).
        f1: F1-score píxel a píxel (None si no hay máscara GT).
        label: Etiqueta descriptiva (ej. dirección de la parcela).
    """

    timestamp: str
    pred_area_m2: float
    manual_area_m2: float
    mape_pct: float
    iou: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    label: str = ""


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Calcula el Intersection over Union entre dos máscaras booleanas.

    Args:
        pred_mask: Máscara predicha (bool 2D).
        gt_mask: Máscara de verdad de campo (bool 2D).

    Returns:
        IoU en [0, 1]. Devuelve 0.0 si la unión es vacía.
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_precision_recall_f1(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> dict[str, float]:
    """Calcula Precision, Recall y F1-score a nivel de píxel.

    Args:
        pred_mask: Máscara predicha (bool 2D).
        gt_mask: Máscara de verdad de campo (bool 2D).

    Returns:
        Dict con claves ``precision``, ``recall``, ``f1``, todas en [0, 1].
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_area_error(pred_area_m2: float, manual_area_m2: float) -> dict[str, float]:
    """Calcula el error de área entre predicción y medición manual.

    Args:
        pred_area_m2: Área predicha en m².
        manual_area_m2: Área manual de referencia en m².

    Returns:
        Dict con:

        - ``mape_pct``: Error porcentual absoluto (%).
        - ``abs_error_m2``: Error absoluto en m².
        - ``rel_error_pct``: Error relativo con signo (%) — positivo = sobredetección.

    Raises:
        ValueError: Si ``manual_area_m2`` es cero o negativo.
    """
    if manual_area_m2 <= 0:
        raise ValueError(
            f"El área manual de referencia debe ser positiva, recibido: {manual_area_m2} m²."
        )

    abs_error = abs(pred_area_m2 - manual_area_m2)
    mape_pct = 100.0 * abs_error / manual_area_m2
    rel_error_pct = 100.0 * (pred_area_m2 - manual_area_m2) / manual_area_m2

    return {
        "mape_pct": round(mape_pct, 2),
        "abs_error_m2": round(abs_error, 2),
        "rel_error_pct": round(rel_error_pct, 2),
    }


def validate(
    pred_area_m2: float,
    manual_area_m2: float,
    pred_mask: np.ndarray | None = None,
    gt_mask: np.ndarray | None = None,
    label: str = "",
) -> ValidationResult:
    """Valida una predicción calculando todas las métricas disponibles.

    Si se proveen ``pred_mask`` y ``gt_mask``, calcula además IoU,
    Precision, Recall y F1. De lo contrario solo calcula el error de área.

    Args:
        pred_area_m2: Área total detectada automáticamente en m².
        manual_area_m2: Área medida manualmente por el usuario en m².
        pred_mask: Máscara booleana predicha ``(H, W)``. Opcional.
        gt_mask: Máscara booleana de verdad de campo ``(H, W)``. Opcional.
        label: Etiqueta descriptiva (ej. dirección). Opcional.

    Returns:
        :class:`ValidationResult` con todas las métricas calculadas.

    Raises:
        ValueError: Si ``manual_area_m2`` ≤ 0.
    """
    area_err = compute_area_error(pred_area_m2, manual_area_m2)

    iou = precision = recall = f1 = None
    if pred_mask is not None and gt_mask is not None:
        iou = round(compute_iou(pred_mask, gt_mask), 4)
        prf = compute_precision_recall_f1(pred_mask, gt_mask)
        precision = round(prf["precision"], 4)
        recall = round(prf["recall"], 4)
        f1 = round(prf["f1"], 4)

    result = ValidationResult(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        pred_area_m2=round(pred_area_m2, 2),
        manual_area_m2=round(manual_area_m2, 2),
        mape_pct=area_err["mape_pct"],
        iou=iou,
        precision=precision,
        recall=recall,
        f1=f1,
        label=label,
    )

    log.info(
        "Validación '%s' | MAPE=%.1f%% | área pred=%.1f m² | área manual=%.1f m²%s",
        label or "(sin etiqueta)",
        result.mape_pct,
        pred_area_m2,
        manual_area_m2,
        f" | IoU={iou:.3f}" if iou is not None else "",
    )
    return result


def log_validation_result(result: ValidationResult, log_file: Path) -> None:
    """Agrega un resultado de validación al log CSV.

    Crea el archivo con encabezado si no existe.

    Args:
        result: Resultado a registrar.
        log_file: Ruta al archivo CSV de log.
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_exists = log_file.exists()
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_LOG_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(asdict(result))

    log.debug("Resultado de validación guardado en %s", log_file)


def load_validation_log(log_file: Path) -> list[dict]:
    """Lee el log CSV de validaciones y devuelve una lista de dicts.

    Args:
        log_file: Ruta al archivo CSV.

    Returns:
        Lista de dicts, uno por validación registrada. Vacío si el archivo
        no existe.
    """
    log_file = Path(log_file)
    if not log_file.exists():
        return []

    rows = []
    with open(log_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convertir campos numéricos
            for field in ("pred_area_m2", "manual_area_m2", "mape_pct",
                          "iou", "precision", "recall", "f1"):
                val = row.get(field, "")
                if val == "" or val is None:
                    row[field] = None
                else:
                    try:
                        row[field] = float(val)
                    except ValueError:
                        row[field] = None
            rows.append(row)

    return rows


def summary_stats(log_file: Path) -> dict[str, float | int | None]:
    """Calcula estadísticas resumidas del historial de validaciones.

    Args:
        log_file: Ruta al archivo CSV de log.

    Returns:
        Dict con ``n_validations``, ``mean_mape_pct``, ``mean_iou``,
        ``mean_f1``. Devuelve ``None`` para métricas sin datos.
    """
    rows = load_validation_log(log_file)
    if not rows:
        return {"n_validations": 0, "mean_mape_pct": None,
                "mean_iou": None, "mean_f1": None}

    mapes = [r["mape_pct"] for r in rows if r["mape_pct"] is not None]
    ious = [r["iou"] for r in rows if r["iou"] is not None]
    f1s = [r["f1"] for r in rows if r["f1"] is not None]

    return {
        "n_validations": len(rows),
        "mean_mape_pct": round(float(np.mean(mapes)), 2) if mapes else None,
        "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
        "mean_f1": round(float(np.mean(f1s)), 4) if f1s else None,
    }
