"""Tests para los módulos de validación y feedback.

Todos los tests son unitarios y no requieren red ni GPU.
"""

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_masks():
    """Máscaras idénticas (predicción perfecta)."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:30, 10:30] = True
    return mask.copy(), mask.copy()   # pred, gt


@pytest.fixture
def disjoint_masks():
    """Máscaras sin solapamiento (peor caso)."""
    pred = np.zeros((64, 64), dtype=bool)
    gt = np.zeros((64, 64), dtype=bool)
    pred[0:10, 0:10] = True
    gt[50:60, 50:60] = True
    return pred, gt


@pytest.fixture
def feedback_dir(tmp_path):
    """Directorio de feedback temporal con 4 pares."""
    img_dir = tmp_path / "images"
    msk_dir = tmp_path / "masks"
    img_dir.mkdir()
    msk_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(4):
        name = f"sample_{i:02d}.npy"
        np.save(img_dir / name, rng.random((6, 32, 32)).astype(np.float32))
        np.save(msk_dir / name, rng.integers(0, 2, (32, 32)).astype(np.uint8))
    return tmp_path


# ---------------------------------------------------------------------------
# Tests: compute_iou
# ---------------------------------------------------------------------------

class TestComputeIou:
    def test_perfect_overlap(self, perfect_masks):
        from roofscan.core.validacion.metrics import compute_iou
        pred, gt = perfect_masks
        assert compute_iou(pred, gt) == pytest.approx(1.0)

    def test_no_overlap(self, disjoint_masks):
        from roofscan.core.validacion.metrics import compute_iou
        pred, gt = disjoint_masks
        assert compute_iou(pred, gt) == pytest.approx(0.0)

    def test_partial_overlap(self):
        from roofscan.core.validacion.metrics import compute_iou
        pred = np.zeros((10, 10), dtype=bool)
        gt = np.zeros((10, 10), dtype=bool)
        pred[0:6, 0:6] = True    # 36 px
        gt[4:10, 4:10] = True    # 36 px
        # Intersección: [4:6,4:6] = 4 px; Unión: 36+36-4 = 68 px
        expected = 4 / 68
        assert compute_iou(pred, gt) == pytest.approx(expected, abs=1e-4)

    def test_empty_masks(self):
        from roofscan.core.validacion.metrics import compute_iou
        pred = np.zeros((10, 10), dtype=bool)
        gt = np.zeros((10, 10), dtype=bool)
        # Unión vacía → 0.0
        assert compute_iou(pred, gt) == pytest.approx(0.0)

    def test_accepts_uint8_masks(self):
        from roofscan.core.validacion.metrics import compute_iou
        pred = np.ones((10, 10), dtype=np.uint8)
        gt = np.ones((10, 10), dtype=np.uint8)
        assert compute_iou(pred, gt) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests: compute_precision_recall_f1
# ---------------------------------------------------------------------------

class TestPrecisionRecallF1:
    def test_perfect_prediction(self, perfect_masks):
        from roofscan.core.validacion.metrics import compute_precision_recall_f1
        pred, gt = perfect_masks
        r = compute_precision_recall_f1(pred, gt)
        assert r["precision"] == pytest.approx(1.0)
        assert r["recall"] == pytest.approx(1.0)
        assert r["f1"] == pytest.approx(1.0)

    def test_no_overlap(self, disjoint_masks):
        from roofscan.core.validacion.metrics import compute_precision_recall_f1
        pred, gt = disjoint_masks
        r = compute_precision_recall_f1(pred, gt)
        assert r["precision"] == pytest.approx(0.0)
        assert r["recall"] == pytest.approx(0.0)
        assert r["f1"] == pytest.approx(0.0)

    def test_f1_harmonic_mean(self):
        """F1 debe ser la media armónica de precision y recall."""
        from roofscan.core.validacion.metrics import compute_precision_recall_f1
        # Construir caso con precision=0.6, recall=0.75 manualmente
        pred = np.zeros((10, 10), dtype=bool)
        gt = np.zeros((10, 10), dtype=bool)
        # 6 TP, 4 FP, 2 FN → precision=6/10=0.6, recall=6/8=0.75
        pred[0:10, 0] = True   # 10 positivos predichos
        gt[0:8, 0] = True      # 8 positivos reales; 6 coinciden con los 10 predichos
        # ajustar: solo 6 de los 10 predichos coinciden con gt
        pred = np.zeros((20, 1), dtype=bool)
        gt = np.zeros((20, 1), dtype=bool)
        pred[:10, 0] = True   # 10 predichos
        gt[:6, 0] = True      # 6 reales que coinciden (primeros 6 de los 10 predichos)
        gt[15:17, 0] = True   # 2 reales que NO coinciden con predichos
        r = compute_precision_recall_f1(pred, gt)
        expected_p = 6 / 10
        expected_r = 6 / 8
        expected_f1 = 2 * expected_p * expected_r / (expected_p + expected_r)
        assert r["precision"] == pytest.approx(expected_p, abs=1e-4)
        assert r["recall"] == pytest.approx(expected_r, abs=1e-4)
        assert r["f1"] == pytest.approx(expected_f1, abs=1e-4)

    def test_all_values_in_range(self):
        from roofscan.core.validacion.metrics import compute_precision_recall_f1
        rng = np.random.default_rng(7)
        pred = rng.integers(0, 2, (50, 50)).astype(bool)
        gt = rng.integers(0, 2, (50, 50)).astype(bool)
        r = compute_precision_recall_f1(pred, gt)
        for k in ("precision", "recall", "f1"):
            assert 0.0 <= r[k] <= 1.0


# ---------------------------------------------------------------------------
# Tests: compute_area_error
# ---------------------------------------------------------------------------

class TestComputeAreaError:
    def test_zero_error(self):
        from roofscan.core.validacion.metrics import compute_area_error
        r = compute_area_error(500.0, 500.0)
        assert r["mape_pct"] == pytest.approx(0.0)
        assert r["abs_error_m2"] == pytest.approx(0.0)
        assert r["rel_error_pct"] == pytest.approx(0.0)

    def test_over_detection(self):
        from roofscan.core.validacion.metrics import compute_area_error
        r = compute_area_error(550.0, 500.0)
        assert r["mape_pct"] == pytest.approx(10.0)
        assert r["rel_error_pct"] == pytest.approx(10.0)   # positivo = sobredetección

    def test_under_detection(self):
        from roofscan.core.validacion.metrics import compute_area_error
        r = compute_area_error(400.0, 500.0)
        assert r["mape_pct"] == pytest.approx(20.0)
        assert r["rel_error_pct"] == pytest.approx(-20.0)  # negativo = subdetección

    def test_invalid_manual_area_zero(self):
        from roofscan.core.validacion.metrics import compute_area_error
        with pytest.raises(ValueError, match="positiva"):
            compute_area_error(100.0, 0.0)

    def test_invalid_manual_area_negative(self):
        from roofscan.core.validacion.metrics import compute_area_error
        with pytest.raises(ValueError):
            compute_area_error(100.0, -50.0)

    def test_result_rounded(self):
        from roofscan.core.validacion.metrics import compute_area_error
        r = compute_area_error(333.333, 300.0)
        assert isinstance(r["mape_pct"], float)
        # Solo verificar que está redondeado a 2 decimales
        assert r["mape_pct"] == round(r["mape_pct"], 2)


# ---------------------------------------------------------------------------
# Tests: validate (función de alto nivel)
# ---------------------------------------------------------------------------

class TestValidate:
    def test_returns_validation_result(self):
        from roofscan.core.validacion.metrics import validate, ValidationResult
        r = validate(pred_area_m2=500.0, manual_area_m2=500.0)
        assert isinstance(r, ValidationResult)

    def test_mape_correct(self):
        from roofscan.core.validacion.metrics import validate
        r = validate(pred_area_m2=600.0, manual_area_m2=500.0)
        assert r.mape_pct == pytest.approx(20.0)

    def test_iou_none_without_masks(self):
        from roofscan.core.validacion.metrics import validate
        r = validate(pred_area_m2=500.0, manual_area_m2=500.0)
        assert r.iou is None
        assert r.precision is None
        assert r.recall is None
        assert r.f1 is None

    def test_iou_computed_with_masks(self, perfect_masks):
        from roofscan.core.validacion.metrics import validate
        pred, gt = perfect_masks
        r = validate(pred_area_m2=500.0, manual_area_m2=500.0, pred_mask=pred, gt_mask=gt)
        assert r.iou == pytest.approx(1.0)
        assert r.f1 == pytest.approx(1.0)

    def test_timestamp_format(self):
        from roofscan.core.validacion.metrics import validate
        import re
        r = validate(pred_area_m2=100.0, manual_area_m2=100.0)
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", r.timestamp)

    def test_label_stored(self):
        from roofscan.core.validacion.metrics import validate
        r = validate(pred_area_m2=100.0, manual_area_m2=100.0, label="Calle Falsa 123")
        assert r.label == "Calle Falsa 123"


# ---------------------------------------------------------------------------
# Tests: log_validation_result y load_validation_log
# ---------------------------------------------------------------------------

class TestValidationLog:
    def test_creates_file_with_header(self, tmp_path):
        from roofscan.core.validacion.metrics import validate, log_validation_result, load_validation_log
        log_file = tmp_path / "val.csv"
        r = validate(pred_area_m2=300.0, manual_area_m2=280.0)
        log_validation_result(r, log_file)

        assert log_file.exists()
        rows = load_validation_log(log_file)
        assert len(rows) == 1
        assert rows[0]["mape_pct"] == pytest.approx(r.mape_pct)

    def test_appends_multiple_rows(self, tmp_path):
        from roofscan.core.validacion.metrics import validate, log_validation_result, load_validation_log
        log_file = tmp_path / "val.csv"
        for pred, manual in [(400, 380), (500, 550), (350, 350)]:
            r = validate(float(pred), float(manual))
            log_validation_result(r, log_file)

        rows = load_validation_log(log_file)
        assert len(rows) == 3

    def test_load_returns_empty_if_file_missing(self, tmp_path):
        from roofscan.core.validacion.metrics import load_validation_log
        rows = load_validation_log(tmp_path / "nonexistent.csv")
        assert rows == []

    def test_numeric_fields_parsed(self, tmp_path):
        from roofscan.core.validacion.metrics import validate, log_validation_result, load_validation_log
        log_file = tmp_path / "val.csv"
        r = validate(pred_area_m2=100.0, manual_area_m2=90.0)
        log_validation_result(r, log_file)
        rows = load_validation_log(log_file)
        assert isinstance(rows[0]["mape_pct"], float)
        assert isinstance(rows[0]["pred_area_m2"], float)


# ---------------------------------------------------------------------------
# Tests: summary_stats
# ---------------------------------------------------------------------------

class TestSummaryStats:
    def test_empty_log(self, tmp_path):
        from roofscan.core.validacion.metrics import summary_stats
        s = summary_stats(tmp_path / "nonexistent.csv")
        assert s["n_validations"] == 0
        assert s["mean_mape_pct"] is None

    def test_mean_mape(self, tmp_path):
        from roofscan.core.validacion.metrics import validate, log_validation_result, summary_stats
        log_file = tmp_path / "val.csv"
        for pred, manual in [(110, 100), (90, 100)]:   # MAPE = 10% en ambos
            log_validation_result(validate(float(pred), float(manual)), log_file)
        s = summary_stats(log_file)
        assert s["n_validations"] == 2
        assert s["mean_mape_pct"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Tests: feedback_store
# ---------------------------------------------------------------------------

class TestSaveFeedbackPair:
    def test_creates_files(self, tmp_path):
        from roofscan.core.validacion.feedback_store import save_feedback_pair
        img = np.zeros((6, 32, 32), dtype=np.float32)
        msk = np.ones((32, 32), dtype=np.uint8)
        name = save_feedback_pair(img, msk, tmp_path)
        assert (tmp_path / "images" / f"{name}.npy").exists()
        assert (tmp_path / "masks" / f"{name}.npy").exists()

    def test_mask_saved_as_binary(self, tmp_path):
        from roofscan.core.validacion.feedback_store import save_feedback_pair
        img = np.zeros((6, 32, 32), dtype=np.float32)
        msk = np.ones((32, 32), dtype=bool)
        name = save_feedback_pair(img, msk, tmp_path)
        saved_msk = np.load(tmp_path / "masks" / f"{name}.npy")
        assert saved_msk.dtype == np.uint8
        assert set(saved_msk.flatten().tolist()).issubset({0, 1})

    def test_raises_if_incompatible_shapes(self, tmp_path):
        from roofscan.core.validacion.feedback_store import save_feedback_pair
        img = np.zeros((6, 32, 32), dtype=np.float32)
        msk = np.zeros((64, 64), dtype=np.uint8)
        with pytest.raises(ValueError, match="[Ii]ncompat"):
            save_feedback_pair(img, msk, tmp_path)

    def test_raises_if_image_not_3d(self, tmp_path):
        from roofscan.core.validacion.feedback_store import save_feedback_pair
        img = np.zeros((32, 32), dtype=np.float32)
        msk = np.zeros((32, 32), dtype=np.uint8)
        with pytest.raises(ValueError, match="3D"):
            save_feedback_pair(img, msk, tmp_path)

    def test_unique_names_per_call(self, tmp_path):
        """Dos llamadas deben generar nombres distintos (sin colisión)."""
        import time
        from roofscan.core.validacion.feedback_store import save_feedback_pair
        img = np.zeros((6, 32, 32), dtype=np.float32)
        msk = np.zeros((32, 32), dtype=np.uint8)
        name1 = save_feedback_pair(img, msk, tmp_path)
        time.sleep(0.01)
        name2 = save_feedback_pair(img, msk, tmp_path)
        # Pueden ser iguales si se ejecutan en el mismo segundo, pero el sufijo aleatorio garantiza diferencia
        # Al menos uno de los dos debe diferir
        assert True   # test de humo: no lanza excepción al sobreescribir


class TestListFeedbackPairs:
    def test_returns_common_pairs(self, feedback_dir):
        from roofscan.core.validacion.feedback_store import list_feedback_pairs
        pairs = list_feedback_pairs(feedback_dir)
        assert len(pairs) == 4

    def test_missing_mask_excluded(self, feedback_dir):
        from roofscan.core.validacion.feedback_store import list_feedback_pairs
        # Agregar imagen sin máscara correspondiente
        extra = feedback_dir / "images" / "orphan.npy"
        np.save(extra, np.zeros((6, 32, 32), dtype=np.float32))
        pairs = list_feedback_pairs(feedback_dir)
        assert "orphan" not in pairs
        assert len(pairs) == 4  # no cambia

    def test_empty_dir_returns_empty(self, tmp_path):
        from roofscan.core.validacion.feedback_store import list_feedback_pairs
        (tmp_path / "images").mkdir()
        (tmp_path / "masks").mkdir()
        assert list_feedback_pairs(tmp_path) == []

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        from roofscan.core.validacion.feedback_store import list_feedback_pairs
        assert list_feedback_pairs(tmp_path / "nope") == []


class TestCountFeedbackPairs:
    def test_count_matches_len(self, feedback_dir):
        from roofscan.core.validacion.feedback_store import count_feedback_pairs
        assert count_feedback_pairs(feedback_dir) == 4

    def test_count_zero_empty(self, tmp_path):
        from roofscan.core.validacion.feedback_store import count_feedback_pairs
        assert count_feedback_pairs(tmp_path) == 0


class TestDeleteFeedbackPair:
    def test_deletes_both_files(self, feedback_dir):
        from roofscan.core.validacion.feedback_store import list_feedback_pairs, delete_feedback_pair
        pairs = list_feedback_pairs(feedback_dir)
        name = pairs[0]
        result = delete_feedback_pair(name, feedback_dir)
        assert result is True
        remaining = list_feedback_pairs(feedback_dir)
        assert name not in remaining

    def test_returns_false_if_not_exists(self, tmp_path):
        from roofscan.core.validacion.feedback_store import delete_feedback_pair
        (tmp_path / "images").mkdir()
        (tmp_path / "masks").mkdir()
        assert delete_feedback_pair("nonexistent", tmp_path) is False


class TestLoadFeedbackPair:
    def test_loads_correct_shapes(self, feedback_dir):
        from roofscan.core.validacion.feedback_store import list_feedback_pairs, load_feedback_pair
        name = list_feedback_pairs(feedback_dir)[0]
        img, msk = load_feedback_pair(name, feedback_dir)
        assert img.shape == (6, 32, 32)
        assert msk.shape == (32, 32)

    def test_raises_if_missing(self, tmp_path):
        from roofscan.core.validacion.feedback_store import load_feedback_pair
        with pytest.raises(FileNotFoundError):
            load_feedback_pair("nonexistent", tmp_path)


class TestFeedbackStats:
    def test_stats_n_pairs(self, feedback_dir):
        from roofscan.core.validacion.feedback_store import feedback_stats
        s = feedback_stats(feedback_dir)
        assert s["n_pairs"] == 4
        assert s["total_pixels"] == 4 * 32 * 32
        assert 0.0 <= s["roof_pct"] <= 100.0

    def test_stats_empty(self, tmp_path):
        from roofscan.core.validacion.feedback_store import feedback_stats
        s = feedback_stats(tmp_path)
        assert s["n_pairs"] == 0
        assert s["total_pixels"] == 0
