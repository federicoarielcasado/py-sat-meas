"""Tests para el módulo de Deep Learning (U-Net, predictor, trainer).

La mayoría no requieren PyTorch instalado realmente — verifican lógica pura
(numpy, paths, validaciones). Los tests marcados con ``@pytest.mark.torch``
sí requieren PyTorch y se saltan si no está disponible.

Correr solo los tests sin dependencias pesadas:
    pytest tests/test_dl.py -m "not torch"

Correr todos (requiere PyTorch + smp instalados):
    pytest tests/test_dl.py
"""

import random
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures globales
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_array():
    """Array float32 (6, 64, 64) normalizado [0, 1]."""
    rng = np.random.default_rng(0)
    return rng.random((6, 64, 64)).astype(np.float32)


@pytest.fixture
def dummy_mask():
    """Máscara booleana (64, 64)."""
    rng = np.random.default_rng(1)
    return rng.integers(0, 2, (64, 64)).astype(bool)


@pytest.fixture
def feedback_dir(tmp_path):
    """Directorio feedback temporal con 6 pares imagen/máscara."""
    img_dir = tmp_path / "images"
    msk_dir = tmp_path / "masks"
    img_dir.mkdir()
    msk_dir.mkdir()

    rng = np.random.default_rng(42)
    for i in range(6):
        name = f"{i:04d}.npy"
        img = rng.random((6, 64, 64)).astype(np.float32)
        msk = rng.integers(0, 2, (64, 64)).astype(np.uint8)
        np.save(img_dir / name, img)
        np.save(msk_dir / name, msk)

    return tmp_path


# ---------------------------------------------------------------------------
# Tests: unet.py — validaciones sin modelo real
# ---------------------------------------------------------------------------

class TestUnetImport:
    def test_build_unet_missing_torch_raises(self):
        """Debe lanzar ImportError descriptivo si torch no está instalado."""
        with patch.dict("sys.modules", {"torch": None}):
            # Re-importar para forzar la lógica de importación
            import importlib
            import roofscan.core.deteccion.dl.unet as unet_mod
            importlib.reload(unet_mod)

            with pytest.raises(ImportError, match="PyTorch"):
                unet_mod.build_unet()

    def test_build_unet_missing_smp_raises(self):
        """Debe lanzar ImportError descriptivo si smp no está instalado."""
        import sys
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"segmentation_models_pytorch": None,
                                         "torch": torch_mock}):
            import importlib
            import roofscan.core.deteccion.dl.unet as unet_mod
            importlib.reload(unet_mod)

            with pytest.raises(ImportError, match="segmentation-models-pytorch"):
                unet_mod.build_unet()


class TestLoadWeightsMissing:
    def test_load_weights_file_not_found(self, tmp_path):
        """Debe lanzar FileNotFoundError con ruta descriptiva."""
        import importlib
        import roofscan.core.deteccion.dl.unet as unet_mod
        importlib.reload(unet_mod)

        fake_model = MagicMock()
        fake_path = tmp_path / "nonexistent.pth"

        with pytest.raises(FileNotFoundError, match="nonexistent.pth"):
            unet_mod.load_weights(fake_model, fake_path)


# ---------------------------------------------------------------------------
# Tests: predictor.py — validaciones de array
# ---------------------------------------------------------------------------

class TestValidateArray:
    def test_rejects_1d_array(self):
        """Array 1D debe lanzar ValueError."""
        from roofscan.core.deteccion.dl.predictor import _validate_array
        with pytest.raises(ValueError, match="3D"):
            _validate_array(np.zeros(10, dtype=np.float32))

    def test_rejects_2d_array(self):
        """Array 2D (H, W) debe lanzar ValueError."""
        from roofscan.core.deteccion.dl.predictor import _validate_array
        with pytest.raises(ValueError, match="3D"):
            _validate_array(np.zeros((64, 64), dtype=np.float32))

    def test_rejects_int_dtype(self):
        """Array entero debe lanzar ValueError solicitando float."""
        from roofscan.core.deteccion.dl.predictor import _validate_array
        with pytest.raises(ValueError, match="float"):
            _validate_array(np.zeros((6, 64, 64), dtype=np.uint8))

    def test_accepts_float32_3d(self, dummy_array):
        """Array float32 3D debe pasar sin excepción."""
        from roofscan.core.deteccion.dl.predictor import _validate_array
        _validate_array(dummy_array)   # no lanza

    def test_accepts_float64_3d(self):
        """Array float64 3D también debe ser aceptado."""
        from roofscan.core.deteccion.dl.predictor import _validate_array
        _validate_array(np.zeros((6, 64, 64), dtype=np.float64))


class TestPadToSize:
    def test_already_correct_size_no_copy(self):
        """Si el array ya tiene el tamaño correcto, se devuelve tal cual."""
        from roofscan.core.deteccion.dl.predictor import _pad_to_size
        arr = np.ones((6, 256, 256), dtype=np.float32)
        out = _pad_to_size(arr, 256)
        assert out is arr   # misma referencia (sin copia)

    def test_pads_smaller_array(self):
        """Array más pequeño se rellena con ceros hasta el tamaño dado."""
        from roofscan.core.deteccion.dl.predictor import _pad_to_size
        arr = np.ones((6, 100, 80), dtype=np.float32)
        out = _pad_to_size(arr, 256)
        assert out.shape == (6, 256, 256)
        assert out[:, :100, :80].sum() == pytest.approx(6 * 100 * 80)
        assert out[:, 100:, :].sum() == 0.0

    def test_original_values_preserved(self):
        """Los valores originales deben estar en la esquina superior izquierda."""
        from roofscan.core.deteccion.dl.predictor import _pad_to_size
        arr = np.arange(6 * 32 * 32, dtype=np.float32).reshape(6, 32, 32)
        out = _pad_to_size(arr, 64)
        np.testing.assert_array_equal(out[:, :32, :32], arr)


class TestGaussianKernel:
    def test_shape(self):
        """El kernel debe tener forma (size, size)."""
        from roofscan.core.deteccion.dl.predictor import _gaussian_kernel
        k = _gaussian_kernel(256)
        assert k.shape == (256, 256)

    def test_center_is_maximum(self):
        """El valor máximo del kernel debe estar en el centro."""
        from roofscan.core.deteccion.dl.predictor import _gaussian_kernel
        k = _gaussian_kernel(64)
        center = k[32, 32]
        assert center == k.max()

    def test_positive_values(self):
        """Todos los valores del kernel deben ser positivos."""
        from roofscan.core.deteccion.dl.predictor import _gaussian_kernel
        k = _gaussian_kernel(32)
        assert (k > 0).all()

    def test_dtype_float32(self):
        from roofscan.core.deteccion.dl.predictor import _gaussian_kernel
        assert _gaussian_kernel(16).dtype == np.float32


# ---------------------------------------------------------------------------
# Tests: trainer.py — lógica pura (sin PyTorch)
# ---------------------------------------------------------------------------

class TestRoofDatasetInit:
    def test_raises_if_images_dir_missing(self, tmp_path):
        """Debe lanzar FileNotFoundError si falta images/."""
        from roofscan.core.deteccion.dl.trainer import RoofDataset
        (tmp_path / "masks").mkdir()
        with pytest.raises(FileNotFoundError, match="images"):
            RoofDataset(tmp_path)

    def test_raises_if_masks_dir_missing(self, tmp_path):
        """Debe lanzar FileNotFoundError si falta masks/."""
        from roofscan.core.deteccion.dl.trainer import RoofDataset
        (tmp_path / "images").mkdir()
        with pytest.raises(FileNotFoundError, match="masks"):
            RoofDataset(tmp_path)

    def test_raises_if_no_pairs(self, tmp_path):
        """Debe lanzar ValueError si no hay pares coincidentes."""
        from roofscan.core.deteccion.dl.trainer import RoofDataset
        (tmp_path / "images").mkdir()
        (tmp_path / "masks").mkdir()
        np.save(tmp_path / "images" / "a.npy", np.zeros((6, 32, 32), dtype=np.float32))
        np.save(tmp_path / "masks" / "b.npy", np.zeros((32, 32), dtype=np.uint8))
        with pytest.raises(ValueError, match="pares"):
            RoofDataset(tmp_path)

    def test_len_matches_pairs(self, feedback_dir):
        """__len__ debe devolver el número de pares encontrados."""
        from roofscan.core.deteccion.dl.trainer import RoofDataset
        ds = RoofDataset(feedback_dir)
        assert len(ds) == 6

    def test_subset_via_indices(self, feedback_dir):
        """El parámetro indices permite seleccionar un subconjunto."""
        from roofscan.core.deteccion.dl.trainer import RoofDataset
        ds = RoofDataset(feedback_dir, indices=[0, 2, 4])
        assert len(ds) == 3


class TestRoofDatasetGetitem:
    def test_getitem_shapes(self, feedback_dir):
        """Los tensores devueltos deben tener forma (6, H, W) e (1, H, W)."""
        import torch
        from roofscan.core.deteccion.dl.trainer import RoofDataset
        ds = RoofDataset(feedback_dir)
        img, msk = ds[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(msk, torch.Tensor)
        assert img.shape[0] == 6
        assert msk.shape[0] == 1
        assert img.shape[1:] == msk.shape[1:]

    def test_mask_binary(self, feedback_dir):
        """Los valores de la máscara deben ser solo 0.0 o 1.0."""
        from roofscan.core.deteccion.dl.trainer import RoofDataset
        ds = RoofDataset(feedback_dir)
        _, msk = ds[0]
        values = msk.unique()
        assert set(values.tolist()).issubset({0.0, 1.0})

    def test_image_dtype_float32(self, feedback_dir):
        """El tensor de imagen debe ser float32."""
        import torch
        from roofscan.core.deteccion.dl.trainer import RoofDataset
        ds = RoofDataset(feedback_dir)
        img, _ = ds[0]
        assert img.dtype == torch.float32

    def test_mask_normalizes_uint8_255(self, tmp_path):
        """Máscaras guardadas como 0/255 deben normalizarse a 0.0/1.0."""
        from roofscan.core.deteccion.dl.trainer import RoofDataset

        img_dir = tmp_path / "images"
        msk_dir = tmp_path / "masks"
        img_dir.mkdir()
        msk_dir.mkdir()

        np.save(img_dir / "0.npy", np.zeros((6, 32, 32), dtype=np.float32))
        msk = np.zeros((32, 32), dtype=np.uint8)
        msk[10:20, 10:20] = 255
        np.save(msk_dir / "0.npy", msk)

        ds = RoofDataset(tmp_path)
        _, msk_t = ds[0]
        assert float(msk_t.max()) == pytest.approx(1.0)
        assert float(msk_t.min()) == pytest.approx(0.0)


class TestAugment:
    def test_augment_preserves_shape(self):
        """La aumentación no debe cambiar la forma del array."""
        from roofscan.core.deteccion.dl.trainer import _augment
        img = np.random.rand(6, 64, 64).astype(np.float32)
        msk = np.random.rand(64, 64).astype(np.float32)
        img_a, msk_a = _augment(img, msk)
        assert img_a.shape == img.shape
        assert msk_a.shape == msk.shape

    def test_augment_preserves_values(self):
        """La aumentación solo reordena píxeles, no cambia valores."""
        from roofscan.core.deteccion.dl.trainer import _augment
        img = np.arange(6 * 16 * 16, dtype=np.float32).reshape(6, 16, 16)
        msk = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
        img_a, msk_a = _augment(img, msk)
        np.testing.assert_array_equal(sorted(img_a.flatten()), sorted(img.flatten()))
        np.testing.assert_array_equal(sorted(msk_a.flatten()), sorted(msk.flatten()))


class TestBceDiceLoss:
    @pytest.mark.torch
    def test_loss_is_scalar(self):
        """La pérdida debe ser un tensor escalar."""
        import torch
        from roofscan.core.deteccion.dl.trainer import _bce_dice_loss
        logits = torch.randn(2, 1, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
        loss = _bce_dice_loss(logits, targets)
        assert loss.ndim == 0

    @pytest.mark.torch
    def test_loss_positive(self):
        """La pérdida debe ser siempre positiva."""
        import torch
        from roofscan.core.deteccion.dl.trainer import _bce_dice_loss
        logits = torch.randn(2, 1, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
        assert _bce_dice_loss(logits, targets).item() > 0

    @pytest.mark.torch
    def test_loss_decreases_perfect_prediction(self):
        """Con predicción perfecta, la pérdida debe ser menor que con aleatoria."""
        import torch
        from roofscan.core.deteccion.dl.trainer import _bce_dice_loss
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()

        # Predicción perfecta: logits muy altos/bajos según targets
        logits_perfect = (targets - 0.5) * 20.0
        logits_random = torch.randn(2, 1, 32, 32)

        loss_perfect = _bce_dice_loss(logits_perfect, targets).item()
        loss_random = _bce_dice_loss(logits_random, targets).item()
        assert loss_perfect < loss_random


# ---------------------------------------------------------------------------
# Tests: fine_tune — solo validaciones de configuración
# ---------------------------------------------------------------------------

class TestFineTuneValidation:
    def test_raises_if_too_few_samples(self, tmp_path):
        """Debe lanzar ValueError si hay menos de 2 muestras."""
        from roofscan.core.deteccion.dl.trainer import fine_tune

        img_dir = tmp_path / "images"
        msk_dir = tmp_path / "masks"
        img_dir.mkdir()
        msk_dir.mkdir()
        # Solo 1 muestra
        np.save(img_dir / "0.npy", np.zeros((6, 32, 32), dtype=np.float32))
        np.save(msk_dir / "0.npy", np.zeros((32, 32), dtype=np.uint8))

        fake_model = MagicMock()
        with pytest.raises(ValueError, match="[Mm]uestra"):
            fine_tune(fake_model, tmp_path, epochs=1)

    def test_raises_if_dataset_missing(self, tmp_path):
        """Debe lanzar FileNotFoundError si dataset_dir no tiene images/."""
        from roofscan.core.deteccion.dl.trainer import fine_tune
        fake_model = MagicMock()
        with pytest.raises(FileNotFoundError):
            fine_tune(fake_model, tmp_path / "nonexistent", epochs=1)


# ---------------------------------------------------------------------------
# Tests de integración @torch — requieren PyTorch real
# ---------------------------------------------------------------------------

@pytest.mark.torch
class TestPredictSingleIntegration:
    """Integración básica de predicción sin tiling."""

    def test_output_shape_matches_input(self):
        """La máscara de salida debe tener la misma H, W que la entrada."""
        import torch
        from roofscan.core.deteccion.dl.predictor import predict_mask

        model = _make_fake_model()
        arr = np.random.rand(6, 64, 64).astype(np.float32)
        mask = predict_mask(model, arr, device="cpu")

        assert mask.shape == (64, 64)
        assert mask.dtype == bool

    def test_output_is_boolean(self):
        """El resultado de predict_mask debe ser booleano."""
        from roofscan.core.deteccion.dl.predictor import predict_mask
        model = _make_fake_model()
        arr = np.random.rand(6, 64, 64).astype(np.float32)
        mask = predict_mask(model, arr, device="cpu")
        assert mask.dtype == bool


@pytest.mark.torch
class TestPredictTiledIntegration:
    """Integración básica de predicción con tiling."""

    def test_tiled_output_shape(self):
        """Con tiling, la máscara de salida debe tener la misma H, W."""
        from roofscan.core.deteccion.dl.predictor import predict_mask
        model = _make_fake_model()
        arr = np.random.rand(6, 400, 300).astype(np.float32)
        mask = predict_mask(model, arr, tile_size=256, overlap=32, device="cpu")
        assert mask.shape == (400, 300)

    def test_proba_range(self):
        """predict_proba debe devolver valores en [0, 1]."""
        from roofscan.core.deteccion.dl.predictor import predict_proba
        model = _make_fake_model()
        arr = np.random.rand(6, 64, 64).astype(np.float32)
        proba = predict_proba(model, arr, device="cpu")
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0
        assert proba.shape == (64, 64)


@pytest.mark.torch
class TestFineTuneIntegration:
    """Integración básica de fine_tune con modelo real minimalista."""

    def test_returns_history_keys(self, feedback_dir):
        """fine_tune debe devolver dict con las claves esperadas."""
        from roofscan.core.deteccion.dl.trainer import fine_tune
        model = _make_fake_model()
        history = fine_tune(
            model, feedback_dir,
            epochs=2, batch_size=2, device="cpu",
        )
        assert "train_loss" in history
        assert "val_loss" in history
        assert "best_epoch" in history
        assert "best_val_loss" in history

    def test_history_length_matches_epochs(self, feedback_dir):
        """El historial debe tener una entrada por época ejecutada."""
        from roofscan.core.deteccion.dl.trainer import fine_tune
        model = _make_fake_model()
        history = fine_tune(
            model, feedback_dir,
            epochs=3, batch_size=2, device="cpu",
            patience=10,   # sin early stopping
        )
        assert len(history["train_loss"]) <= 3
        assert len(history["val_loss"]) <= 3

    def test_saves_model_file(self, feedback_dir, tmp_path):
        """fine_tune debe guardar el mejor modelo en save_path."""
        from roofscan.core.deteccion.dl.trainer import fine_tune
        model = _make_fake_model()
        save_path = tmp_path / "test_model.pth"
        fine_tune(
            model, feedback_dir,
            epochs=1, batch_size=2, device="cpu",
            save_path=save_path,
        )
        assert save_path.exists()
        assert save_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Helpers de test
# ---------------------------------------------------------------------------

def _make_fake_model():
    """Crea un modelo PyTorch mínimo para tests (1 conv, 6 → 1 canales)."""
    import torch
    import torch.nn as nn

    class _FakeUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(6, 1, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    return _FakeUNet()
