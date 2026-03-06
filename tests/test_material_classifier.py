"""Tests para roofscan.core.deteccion.dl.material_classifier.

Tests sin dependencias externas (siempre corren):
    - extract_spectral_stats: extracción de features de array sintético.
    - extract_roof_patch: extracción de parche espacial.
    - _spectral_rule_probs: reglas espectrales para cada material.
    - classify_roof_materials: API principal con fallback espectral.
    - MATERIAL_LABELS / MATERIAL_IDX: consistencia de constantes.

Tests con PyTorch (@pytest.mark.torch — saltan si PyTorch no está):
    - build_material_mlp: construcción del MLP y forward pass.
    - build_material_cnn: construcción de la CNN y forward pass.
    - classify_roof_materials con mlp_model: clasificación con MLP.

Ejecutar sin PyTorch:
    pytest tests/test_material_classifier.py -m "not torch"

Ejecutar todos:
    pytest tests/test_material_classifier.py
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures comunes
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_array():
    """Array Sentinel-2 sintético (6, 100, 100) en rango [0, 1]."""
    rng = np.random.default_rng(42)
    arr = rng.uniform(0.0, 1.0, (6, 100, 100)).astype(np.float32)
    return arr


@pytest.fixture()
def synthetic_transform():
    """Transformada affine compatible con array 100×100 en EPSG:32721."""
    from rasterio.transform import from_bounds
    # bbox 100×100 px a 10 m/px = 1000×1000 m
    return from_bounds(0.0, 0.0, 1000.0, 1000.0, 100, 100)


@pytest.fixture()
def roof_polygon():
    """Polígono rectangular de 50×30 m en el centro del array sintético."""
    from shapely.geometry import box
    return box(450, 450, 550, 500)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

def test_material_labels_count():
    from roofscan.core.deteccion.dl.material_classifier import MATERIAL_LABELS, N_CLASSES
    assert len(MATERIAL_LABELS) == N_CLASSES == 4


def test_material_idx_inverse():
    from roofscan.core.deteccion.dl.material_classifier import MATERIAL_LABELS, MATERIAL_IDX
    for i, label in enumerate(MATERIAL_LABELS):
        assert MATERIAL_IDX[label] == i


def test_n_spectral_features():
    from roofscan.core.deteccion.dl.material_classifier import N_SPECTRAL_FEATURES, N_BANDS
    # 4 stats × 6 bandas + 5 índices
    assert N_SPECTRAL_FEATURES == N_BANDS * 4 + 5 == 29


# ---------------------------------------------------------------------------
# extract_spectral_stats
# ---------------------------------------------------------------------------

def test_extract_spectral_stats_shape(synthetic_array, synthetic_transform, roof_polygon):
    from roofscan.core.deteccion.dl.material_classifier import (
        extract_spectral_stats, N_SPECTRAL_FEATURES,
    )
    feats = extract_spectral_stats(synthetic_array, synthetic_transform, roof_polygon)
    assert feats is not None
    assert feats.shape == (N_SPECTRAL_FEATURES,)
    assert feats.dtype == np.float32


def test_extract_spectral_stats_dtype(synthetic_array, synthetic_transform, roof_polygon):
    from roofscan.core.deteccion.dl.material_classifier import extract_spectral_stats
    feats = extract_spectral_stats(synthetic_array, synthetic_transform, roof_polygon)
    assert feats is not None
    assert np.issubdtype(feats.dtype, np.floating)
    assert not np.any(np.isnan(feats))


def test_extract_spectral_stats_tiny_polygon(synthetic_array, synthetic_transform):
    """Polígono más pequeño que un píxel → usa el centroide."""
    from shapely.geometry import box
    from roofscan.core.deteccion.dl.material_classifier import (
        extract_spectral_stats, N_SPECTRAL_FEATURES,
    )
    tiny = box(500, 500, 501, 501)  # 1×1 m, sub-pixel a 10 m/px
    feats = extract_spectral_stats(synthetic_array, synthetic_transform, tiny)
    # Puede ser None si el centroide cae fuera, pero si cae dentro debe tener forma correcta
    if feats is not None:
        assert feats.shape == (N_SPECTRAL_FEATURES,)


def test_extract_spectral_stats_outside_returns_none(synthetic_array, synthetic_transform):
    """Polígono completamente fuera de la imagen → retorna None."""
    from shapely.geometry import box
    from roofscan.core.deteccion.dl.material_classifier import extract_spectral_stats
    outside = box(5000, 5000, 6000, 6000)
    feats = extract_spectral_stats(synthetic_array, synthetic_transform, outside)
    assert feats is None


# ---------------------------------------------------------------------------
# extract_roof_patch
# ---------------------------------------------------------------------------

def test_extract_roof_patch_shape(synthetic_array, synthetic_transform, roof_polygon):
    from roofscan.core.deteccion.dl.material_classifier import extract_roof_patch, N_BANDS
    patch = extract_roof_patch(synthetic_array, synthetic_transform, roof_polygon, patch_size=32)
    assert patch is not None
    assert patch.shape == (N_BANDS, 32, 32)
    assert patch.dtype == np.float32


def test_extract_roof_patch_outside_returns_none(synthetic_array, synthetic_transform):
    """Centroide fuera de la imagen → None."""
    from shapely.geometry import box
    from roofscan.core.deteccion.dl.material_classifier import extract_roof_patch
    outside = box(5000, 5000, 6000, 6000)
    patch = extract_roof_patch(synthetic_array, synthetic_transform, outside)
    assert patch is None


def test_extract_roof_patch_edge_pads(synthetic_transform):
    """Parche en el borde de la imagen → se rellena con ceros."""
    from shapely.geometry import box
    from roofscan.core.deteccion.dl.material_classifier import extract_roof_patch, N_BANDS
    arr = np.ones((N_BANDS, 100, 100), dtype=np.float32)
    corner_poly = box(0, 0, 30, 30)  # esquina inferior izquierda
    patch = extract_roof_patch(arr, synthetic_transform, corner_poly, patch_size=32)
    assert patch is not None
    assert patch.shape == (N_BANDS, 32, 32)


# ---------------------------------------------------------------------------
# _spectral_rule_probs
# ---------------------------------------------------------------------------

def test_spectral_rules_sum_to_one():
    from roofscan.core.deteccion.dl.material_classifier import (
        _spectral_rule_probs, N_SPECTRAL_FEATURES,
    )
    feats = np.zeros(N_SPECTRAL_FEATURES, dtype=np.float32)
    probs = _spectral_rule_probs(feats)
    assert abs(probs.sum() - 1.0) < 1e-5


def test_spectral_rules_zinc_signature():
    """Firma espectral típica de zinc corrugado: NDBI alto, B11 alto."""
    from roofscan.core.deteccion.dl.material_classifier import (
        _spectral_rule_probs, N_SPECTRAL_FEATURES, MATERIAL_IDX,
    )
    feats = np.zeros(N_SPECTRAL_FEATURES, dtype=np.float32)
    # Simular zinc: B11 (posición 4) y B12 (posición 5) altos, NDBI alto
    # means: posiciones 0–5
    feats[4] = 0.8   # mean_b11 alto
    feats[5] = 0.75  # mean_b12 alto
    feats[24] = -0.1  # ndvi bajo (no vegetación)
    feats[25] = 0.2   # ndbi alto → construido metálico
    feats[27] = 1.05  # ratio b11/b12 ≈ 1
    probs = _spectral_rule_probs(feats)
    assert probs[MATERIAL_IDX["zinc_corrugado"]] == probs.max()


def test_spectral_rules_incomplete_signature():
    """Firma espectral de construcción incompleta: NDVI alto."""
    from roofscan.core.deteccion.dl.material_classifier import (
        _spectral_rule_probs, N_SPECTRAL_FEATURES, MATERIAL_IDX,
    )
    feats = np.zeros(N_SPECTRAL_FEATURES, dtype=np.float32)
    feats[24] = 0.5   # ndvi muy alto → vegetación / sin techo
    probs = _spectral_rule_probs(feats)
    assert probs[MATERIAL_IDX["construccion_incompleta"]] == probs.max()


def test_spectral_rules_ceramica_signature():
    """Firma de tejas cerámicas: ratio B04/B03 elevado."""
    from roofscan.core.deteccion.dl.material_classifier import (
        _spectral_rule_probs, N_SPECTRAL_FEATURES, MATERIAL_IDX,
    )
    feats = np.zeros(N_SPECTRAL_FEATURES, dtype=np.float32)
    feats[24] = -0.05   # ndvi bajo
    feats[25] = -0.05   # ndbi ligeramente negativo
    feats[28] = 1.4     # ratio B04/B03 alto → perfil rojizo
    probs = _spectral_rule_probs(feats)
    assert probs[MATERIAL_IDX["tejas_ceramica"]] == probs.max()


# ---------------------------------------------------------------------------
# classify_roof_materials (sin modelos DL — fallback espectral)
# ---------------------------------------------------------------------------

def test_classify_roof_materials_no_models(synthetic_array, synthetic_transform):
    """Sin modelos: debe clasificar usando reglas espectrales."""
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import box
    from roofscan.core.deteccion.dl.material_classifier import (
        classify_roof_materials, MATERIAL_LABELS,
    )

    polys = [box(300 + i * 100, 300, 350 + i * 100, 350) for i in range(3)]
    gdf = geopandas.GeoDataFrame(
        {"area_m2": [2500.0, 2500.0, 2500.0]},
        geometry=polys,
        crs="EPSG:32721",
    )
    image_data = {"array": synthetic_array, "transform": synthetic_transform}

    result = classify_roof_materials(image_data, gdf)

    assert "material" in result.columns
    assert "material_score" in result.columns
    assert "material_method" in result.columns
    assert len(result) == 3
    assert all(m in MATERIAL_LABELS for m in result["material"])
    assert all(result["material_method"] == "spectral_rules")
    assert all(0.0 <= s <= 1.0 for s in result["material_score"])


def test_classify_roof_materials_preserves_rows(synthetic_array, synthetic_transform):
    """El número de filas del GDF debe ser idéntico antes y después."""
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import box
    from roofscan.core.deteccion.dl.material_classifier import classify_roof_materials

    polys = [box(200 + i * 50, 200, 240 + i * 50, 240) for i in range(5)]
    gdf = geopandas.GeoDataFrame({"area_m2": [1000.0] * 5}, geometry=polys, crs="EPSG:32721")
    image_data = {"array": synthetic_array, "transform": synthetic_transform}

    result = classify_roof_materials(image_data, gdf)
    assert len(result) == len(gdf)


def test_classify_roof_materials_bad_array_raises(synthetic_transform):
    """Array con banda incorrecta debe lanzar ValueError."""
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import box
    from roofscan.core.deteccion.dl.material_classifier import classify_roof_materials

    bad_array = np.ones((3, 50, 50), dtype=np.float32)  # 3 bandas, no 6
    gdf = geopandas.GeoDataFrame({"area_m2": [100.0]}, geometry=[box(10, 10, 20, 20)])
    image_data = {"array": bad_array, "transform": synthetic_transform}

    with pytest.raises(ValueError, match="6 bandas"):
        classify_roof_materials(image_data, gdf)


# ---------------------------------------------------------------------------
# Tests con PyTorch (@pytest.mark.torch)
# ---------------------------------------------------------------------------

@pytest.mark.torch
def test_build_material_mlp_forward():
    """MLP: construcción y forward pass con batch de 4 muestras."""
    from roofscan.core.deteccion.dl.material_classifier import (
        build_material_mlp, N_SPECTRAL_FEATURES, N_CLASSES,
    )
    import torch

    model = build_material_mlp(device="cpu")
    model.eval()

    x = torch.randn(4, N_SPECTRAL_FEATURES)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (4, N_CLASSES)
    assert not torch.isnan(out).any()


@pytest.mark.torch
def test_build_material_cnn_forward():
    """CNN: construcción y forward pass con batch de 2 parches."""
    from roofscan.core.deteccion.dl.material_classifier import (
        build_material_cnn, N_BANDS, N_CLASSES,
    )
    import torch

    model = build_material_cnn(patch_size=32, device="cpu")
    model.eval()

    x = torch.randn(2, N_BANDS, 32, 32)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, N_CLASSES)
    assert not torch.isnan(out).any()


@pytest.mark.torch
def test_classify_with_mlp_model(synthetic_array, synthetic_transform):
    """classify_roof_materials con MLP en modo eval: method debe ser 'mlp'."""
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import box
    from roofscan.core.deteccion.dl.material_classifier import (
        build_material_mlp, classify_roof_materials,
    )

    model = build_material_mlp(device="cpu")
    model.eval()

    polys = [box(300, 300, 380, 380), box(500, 300, 580, 380)]
    gdf = geopandas.GeoDataFrame(
        {"area_m2": [6400.0, 6400.0]},
        geometry=polys,
        crs="EPSG:32721",
    )
    image_data = {"array": synthetic_array, "transform": synthetic_transform}

    result = classify_roof_materials(image_data, gdf, mlp_model=model)

    assert all(result["material_method"] == "mlp")
    assert all(0.0 <= s <= 1.0 for s in result["material_score"])


@pytest.mark.torch
def test_save_load_weights(tmp_path):
    """Guardar y recargar pesos del MLP: state_dict debe coincidir."""
    from roofscan.core.deteccion.dl.material_classifier import (
        build_material_mlp, save_weights, load_weights,
    )
    import torch

    model = build_material_mlp(device="cpu")
    path = tmp_path / "material_mlp.pth"
    save_weights(model, path)

    assert path.exists()

    model2 = build_material_mlp(device="cpu")
    model2 = load_weights(model2, path, device="cpu")

    for (k1, v1), (k2, v2) in zip(
        model.state_dict().items(), model2.state_dict().items()
    ):
        assert k1 == k2
        assert torch.allclose(v1, v2)


@pytest.mark.torch
def test_load_weights_missing_file_raises(tmp_path):
    """load_weights con archivo inexistente debe lanzar FileNotFoundError."""
    from roofscan.core.deteccion.dl.material_classifier import (
        build_material_mlp, load_weights,
    )
    model = build_material_mlp(device="cpu")
    with pytest.raises(FileNotFoundError):
        load_weights(model, tmp_path / "no_existe.pth")
