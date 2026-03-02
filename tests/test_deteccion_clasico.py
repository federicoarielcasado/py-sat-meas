"""Tests del motor clásico de detección: índices espectrales, morfología y calculo."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def s2_array_normalized() -> np.ndarray:
    """Array sintético normalizado de 6 bandas S2 (B02-B03-B04-B08-B11-B12), 30x30 px."""
    rng = np.random.default_rng(7)
    # Valores de reflectancia razonables [0.0, 0.7]
    return rng.uniform(0.05, 0.70, (6, 30, 30)).astype(np.float32)


@pytest.fixture()
def s2_with_roofs() -> np.ndarray:
    """Array con una zona central de 10x10 px simulando un techo (NDBI alto, NDVI bajo)."""
    arr = np.full((6, 30, 30), 0.15, dtype=np.float32)  # fondo: vegetación media
    # Zona de techo (centro 10x10): SWIR alto, NIR bajo, Rojo alto
    arr[2, 10:20, 10:20] = 0.35  # B04 (Rojo) ↑
    arr[3, 10:20, 10:20] = 0.18  # B08 (NIR) bajo → NDVI bajo
    arr[4, 10:20, 10:20] = 0.45  # B11 (SWIR1) ↑ → NDBI alto
    return arr


@pytest.fixture()
def binary_mask_with_noise() -> np.ndarray:
    """Máscara binaria con un objeto grande y varios píxeles de ruido aislados."""
    mask = np.zeros((30, 30), dtype=bool)
    mask[10:20, 10:20] = True   # objeto real: 100 px
    mask[0, 0] = True            # ruido: 1 px
    mask[2, 5] = True            # ruido: 1 px
    return mask


@pytest.fixture()
def labeled_array() -> np.ndarray:
    """Mapa de etiquetas con 2 objetos: uno de 100 px y otro de 25 px."""
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[0:10, 0:10] = 1    # objeto 1: 100 px
    labels[20:25, 20:25] = 2  # objeto 2: 25 px
    return labels


# ---------------------------------------------------------------------------
# Tests: spectral_indices
# ---------------------------------------------------------------------------

class TestSpectralIndices:
    def test_compute_ndvi_shape(self, s2_array_normalized):
        from roofscan.core.deteccion.clasico.spectral_indices import compute_ndvi
        ndvi = compute_ndvi(s2_array_normalized)
        assert ndvi.shape == (30, 30)

    def test_compute_ndvi_range(self, s2_array_normalized):
        from roofscan.core.deteccion.clasico.spectral_indices import compute_ndvi
        ndvi = compute_ndvi(s2_array_normalized)
        valid = ndvi[~np.isnan(ndvi)]
        assert valid.min() >= -1.0
        assert valid.max() <= 1.0

    def test_compute_ndvi_dtype_float32(self, s2_array_normalized):
        from roofscan.core.deteccion.clasico.spectral_indices import compute_ndvi
        assert compute_ndvi(s2_array_normalized).dtype == np.float32

    def test_compute_ndbi_shape(self, s2_array_normalized):
        from roofscan.core.deteccion.clasico.spectral_indices import compute_ndbi
        assert compute_ndbi(s2_array_normalized).shape == (30, 30)

    def test_compute_ndwi_shape(self, s2_array_normalized):
        from roofscan.core.deteccion.clasico.spectral_indices import compute_ndwi
        assert compute_ndwi(s2_array_normalized).shape == (30, 30)

    def test_ndvi_nan_propagation(self):
        from roofscan.core.deteccion.clasico.spectral_indices import compute_ndvi
        arr = np.ones((6, 5, 5), dtype=np.float32) * 0.3
        arr[3, 2, 2] = np.nan  # NIR NaN → NDVI NaN
        ndvi = compute_ndvi(arr)
        assert np.isnan(ndvi[2, 2])

    def test_zero_denominator_returns_nan(self):
        from roofscan.core.deteccion.clasico.spectral_indices import compute_ndvi
        # NIR = Red = 0 → denominador 0 → NaN
        arr = np.full((6, 5, 5), 0.3, dtype=np.float32)
        arr[2, :, :] = 0.0  # Red = 0
        arr[3, :, :] = 0.0  # NIR = 0
        ndvi = compute_ndvi(arr)
        assert np.isnan(ndvi).all()

    def test_detect_roofs_returns_required_keys(self, s2_array_normalized):
        from roofscan.core.deteccion.clasico.spectral_indices import detect_roofs
        result = detect_roofs(s2_array_normalized)
        for key in ("mask", "ndvi", "ndbi", "ndwi", "coverage_pct", "detection_config"):
            assert key in result

    def test_detect_roofs_mask_is_bool(self, s2_array_normalized):
        from roofscan.core.deteccion.clasico.spectral_indices import detect_roofs
        result = detect_roofs(s2_array_normalized)
        assert result["mask"].dtype == bool

    def test_detect_roofs_coverage_between_0_100(self, s2_array_normalized):
        from roofscan.core.deteccion.clasico.spectral_indices import detect_roofs
        result = detect_roofs(s2_array_normalized)
        assert 0.0 <= result["coverage_pct"] <= 100.0

    def test_detect_roofs_detects_synthetic_roof(self, s2_with_roofs):
        from roofscan.core.deteccion.clasico.spectral_indices import detect_roofs, DetectionConfig
        cfg = DetectionConfig(ndvi_max=0.25, ndbi_min=-0.10, ndwi_max=0.20)
        result = detect_roofs(s2_with_roofs, config=cfg)
        # La zona central debe tener mayor densidad de detección que el fondo
        center = result["mask"][10:20, 10:20].sum()
        corner = result["mask"][0:5, 0:5].sum()
        assert center > corner

    def test_detect_roofs_raises_non_float_array(self):
        from roofscan.core.deteccion.clasico.spectral_indices import detect_roofs
        arr = np.ones((6, 10, 10), dtype=np.uint16)
        with pytest.raises(ValueError, match="float"):
            detect_roofs(arr)

    def test_detect_roofs_raises_too_few_bands(self):
        from roofscan.core.deteccion.clasico.spectral_indices import detect_roofs
        arr = np.ones((3, 10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="6 bandas"):
            detect_roofs(arr)


# ---------------------------------------------------------------------------
# Tests: morphology
# ---------------------------------------------------------------------------

class TestMorphology:
    def test_clean_mask_removes_noise(self, binary_mask_with_noise):
        from roofscan.core.deteccion.clasico.morphology import clean_mask, MorphologyConfig
        cfg = MorphologyConfig(min_area_px=10)
        clean = clean_mask(binary_mask_with_noise, config=cfg)
        # Los píxeles de ruido en (0,0) y (2,5) deben desaparecer
        assert not clean[0, 0]
        assert not clean[2, 5]

    def test_clean_mask_preserves_large_object(self, binary_mask_with_noise):
        from roofscan.core.deteccion.clasico.morphology import clean_mask, MorphologyConfig
        cfg = MorphologyConfig(min_area_px=10, close_radius=0, open_radius=0, fill_holes=False)
        clean = clean_mask(binary_mask_with_noise, config=cfg)
        assert clean[15, 15]  # centro del objeto grande

    def test_clean_mask_output_is_bool(self, binary_mask_with_noise):
        from roofscan.core.deteccion.clasico.morphology import clean_mask
        clean = clean_mask(binary_mask_with_noise)
        assert clean.dtype == bool

    def test_clean_mask_fills_holes(self):
        from roofscan.core.deteccion.clasico.morphology import clean_mask, MorphologyConfig
        # Anillo con hueco interior
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        mask[8:12, 8:12] = False  # agujero
        cfg = MorphologyConfig(fill_holes=True, close_radius=0, open_radius=0, min_area_px=1)
        clean = clean_mask(mask, config=cfg)
        assert clean[10, 10]  # el agujero debe estar relleno

    def test_clean_mask_raises_on_3d(self):
        from roofscan.core.deteccion.clasico.morphology import clean_mask
        with pytest.raises(ValueError, match="2D"):
            clean_mask(np.ones((3, 10, 10), dtype=bool))

    def test_label_roofs_counts_objects(self):
        from roofscan.core.deteccion.clasico.morphology import label_roofs
        mask = np.zeros((20, 20), dtype=bool)
        mask[0:5, 0:5] = True    # objeto 1
        mask[10:15, 10:15] = True  # objeto 2
        _, n = label_roofs(mask)
        assert n == 2

    def test_run_morphology_returns_required_keys(self, binary_mask_with_noise):
        from roofscan.core.deteccion.clasico.morphology import run_morphology
        result = run_morphology(binary_mask_with_noise)
        for key in ("mask_clean", "labels", "n_roofs", "morphology_config"):
            assert key in result

    def test_run_morphology_labels_dtype_int32(self, binary_mask_with_noise):
        from roofscan.core.deteccion.clasico.morphology import run_morphology
        result = run_morphology(binary_mask_with_noise)
        assert result["labels"].dtype == np.int32


# ---------------------------------------------------------------------------
# Tests: area_calculator
# ---------------------------------------------------------------------------

class TestAreaCalculator:
    def test_calculate_areas_returns_list(self, labeled_array):
        from roofscan.core.calculo.area_calculator import calculate_areas
        areas = calculate_areas(labeled_array, resolution_m=10.0)
        assert isinstance(areas, list)

    def test_calculate_areas_correct_count(self, labeled_array):
        from roofscan.core.calculo.area_calculator import calculate_areas
        areas = calculate_areas(labeled_array, resolution_m=10.0)
        assert len(areas) == 2

    def test_calculate_areas_correct_m2(self, labeled_array):
        from roofscan.core.calculo.area_calculator import calculate_areas
        areas = calculate_areas(labeled_array, resolution_m=10.0)
        # Objeto 1: 100 px × 100 m²/px = 10 000 m²
        # Objeto 2: 25 px × 100 m²/px = 2 500 m²
        areas_m2 = {r["id"]: r["area_m2"] for r in areas}
        assert areas_m2[1] == pytest.approx(10_000.0)
        assert areas_m2[2] == pytest.approx(2_500.0)

    def test_calculate_areas_sorted_by_area_desc(self, labeled_array):
        from roofscan.core.calculo.area_calculator import calculate_areas
        areas = calculate_areas(labeled_array, resolution_m=10.0)
        # Deben estar ordenados de mayor a menor
        assert areas[0]["area_m2"] >= areas[1]["area_m2"]

    def test_calculate_areas_has_centroid(self, labeled_array):
        from roofscan.core.calculo.area_calculator import calculate_areas
        areas = calculate_areas(labeled_array, resolution_m=10.0)
        for r in areas:
            assert "centroid_px" in r
            assert len(r["centroid_px"]) == 2

    def test_calculate_areas_empty_labels(self):
        from roofscan.core.calculo.area_calculator import calculate_areas
        empty = np.zeros((20, 20), dtype=np.int32)
        areas = calculate_areas(empty, resolution_m=10.0)
        assert areas == []

    def test_calculate_areas_min_area_filter(self, labeled_array):
        from roofscan.core.calculo.area_calculator import calculate_areas
        # Filtrar objetos con menos de 3 000 m² → solo queda el objeto grande
        areas = calculate_areas(labeled_array, resolution_m=10.0, min_area_m2=3_000.0)
        assert len(areas) == 1
        assert areas[0]["id"] == 1

    def test_calculate_areas_invalid_resolution(self, labeled_array):
        from roofscan.core.calculo.area_calculator import calculate_areas
        with pytest.raises(ValueError, match="resolution_m"):
            calculate_areas(labeled_array, resolution_m=0.0)

    def test_total_covered_area(self, labeled_array):
        from roofscan.core.calculo.area_calculator import calculate_areas, total_covered_area_m2
        areas = calculate_areas(labeled_array, resolution_m=10.0)
        total = total_covered_area_m2(areas)
        assert total == pytest.approx(12_500.0)
