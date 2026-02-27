"""Tests del módulo de preprocesamiento: reprojector, cloud_mask, normalizer y pipeline."""

from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures compartidas
# ---------------------------------------------------------------------------

@pytest.fixture()
def spectral_array_utm() -> np.ndarray:
    """Array espectral sintético: 6 bandas, 40x40 px, valores DN S2 (0-10000)."""
    rng = np.random.default_rng(42)
    arr = rng.integers(500, 8000, size=(6, 40, 40), dtype=np.uint16)
    return arr


@pytest.fixture()
def scl_array_clean() -> np.ndarray:
    """Banda SCL sin nubes (clase 5 = suelo desnudo en toda la imagen)."""
    return np.full((40, 40), 5, dtype=np.uint8)


@pytest.fixture()
def scl_array_half_cloud() -> np.ndarray:
    """Banda SCL con mitad de la imagen nubosa (clase 9)."""
    arr = np.full((40, 40), 5, dtype=np.uint8)
    arr[:20, :] = 9  # mitad superior = nubes alta probabilidad
    return arr


@pytest.fixture()
def data_utm(tmp_path: Path) -> dict:
    """Dict estilo load_geotiff: raster sintético en EPSG:32720, 10 m/px."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    width = height = 40
    west, south, east, north = 343000, 6172000, 343400, 6172400
    transform = from_bounds(west, south, east, north, width, height)

    rng = np.random.default_rng(0)
    array = rng.integers(500, 8000, (6, height, width), dtype=np.uint16)

    return {
        "array": array,
        "crs": "EPSG:32720",
        "transform": transform,
        "bounds": type("B", (), {"left": west, "bottom": south, "right": east, "top": north})(),
        "resolution_m": 10.0,
        "nodata": 0,
        "count": 6,
        "dtype": "uint16",
        "sensor": "Sentinel-2",
        "filepath": str(tmp_path / "fake.tif"),
    }


@pytest.fixture()
def data_wgs84(tmp_path: Path) -> dict:
    """Dict estilo load_geotiff: raster sintético en EPSG:4326 (grados)."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    width = height = 40
    west, south, east, north = -59.15, -34.60, -59.05, -34.53
    transform = from_bounds(west, south, east, north, width, height)

    rng = np.random.default_rng(1)
    array = rng.integers(500, 8000, (6, height, width), dtype=np.uint16)

    return {
        "array": array,
        "crs": "EPSG:4326",
        "transform": transform,
        "bounds": type("B", (), {"left": west, "bottom": south, "right": east, "top": north})(),
        "resolution_m": None,
        "nodata": 0,
        "count": 6,
        "dtype": "uint16",
        "sensor": "desconocido",
        "filepath": str(tmp_path / "fake_wgs84.tif"),
    }


# ---------------------------------------------------------------------------
# Tests: reprojector
# ---------------------------------------------------------------------------

class TestReprojector:
    def test_same_crs_returns_unchanged(self, data_utm):
        from roofscan.core.preproceso.reprojector import reproject_to_crs

        result = reproject_to_crs(data_utm, target_crs="EPSG:32720")
        # No debe copiar el array
        assert result is data_utm

    def test_reproject_wgs84_to_utm(self, data_wgs84):
        from roofscan.core.preproceso.reprojector import reproject_to_crs

        result = reproject_to_crs(data_wgs84, target_crs="EPSG:32720")
        assert "32720" in result["crs"]

    def test_reproject_preserves_band_count(self, data_wgs84):
        from roofscan.core.preproceso.reprojector import reproject_to_crs

        result = reproject_to_crs(data_wgs84, target_crs="EPSG:32720")
        assert result["array"].shape[0] == data_wgs84["array"].shape[0]

    def test_reproject_resolution_is_metric(self, data_wgs84):
        from roofscan.core.preproceso.reprojector import reproject_to_crs

        result = reproject_to_crs(data_wgs84, target_crs="EPSG:32720")
        assert result["resolution_m"] is not None
        assert result["resolution_m"] > 0

    def test_missing_crs_raises_value_error(self, data_utm):
        from roofscan.core.preproceso.reprojector import reproject_to_crs

        data = dict(data_utm)
        data["crs"] = None
        with pytest.raises(ValueError, match="CRS"):
            reproject_to_crs(data)

    def test_is_metric_crs_utm(self):
        from roofscan.core.preproceso.reprojector import is_metric_crs
        assert is_metric_crs("EPSG:32720") is True

    def test_is_metric_crs_wgs84_false(self):
        from roofscan.core.preproceso.reprojector import is_metric_crs
        assert is_metric_crs("EPSG:4326") is False


# ---------------------------------------------------------------------------
# Tests: cloud_mask
# ---------------------------------------------------------------------------

class TestCloudMask:
    def test_apply_no_clouds_all_valid(self, spectral_array_utm, scl_array_clean):
        from roofscan.core.preproceso.cloud_mask import apply_cloud_mask

        masked, valid_mask = apply_cloud_mask(spectral_array_utm, scl_array_clean)
        assert valid_mask.all(), "Sin nubes, todos los píxeles deben ser válidos"
        assert not np.isnan(masked).any(), "Sin nubes, no debe haber NaN"

    def test_apply_half_cloud_masks_correctly(self, spectral_array_utm, scl_array_half_cloud):
        from roofscan.core.preproceso.cloud_mask import apply_cloud_mask

        masked, valid_mask = apply_cloud_mask(spectral_array_utm, scl_array_half_cloud)
        # Mitad superior debe ser inválida
        assert not valid_mask[:20, :].any()
        assert valid_mask[20:, :].all()

    def test_apply_cloud_mask_nodata_is_nan(self, spectral_array_utm, scl_array_half_cloud):
        from roofscan.core.preproceso.cloud_mask import apply_cloud_mask

        masked, _ = apply_cloud_mask(spectral_array_utm, scl_array_half_cloud, nodata_value=np.nan)
        assert np.isnan(masked[:, :20, :]).all()

    def test_apply_cloud_mask_output_dtype_float32(self, spectral_array_utm, scl_array_clean):
        from roofscan.core.preproceso.cloud_mask import apply_cloud_mask

        masked, _ = apply_cloud_mask(spectral_array_utm, scl_array_clean)
        assert masked.dtype == np.float32

    def test_dimension_mismatch_raises_value_error(self, spectral_array_utm):
        from roofscan.core.preproceso.cloud_mask import apply_cloud_mask

        bad_scl = np.full((20, 20), 5, dtype=np.uint8)  # dimensiones distintas
        with pytest.raises(ValueError, match="dimensiones"):
            apply_cloud_mask(spectral_array_utm, bad_scl)

    def test_compute_cloud_coverage_no_clouds(self, scl_array_clean):
        from roofscan.core.preproceso.cloud_mask import compute_cloud_coverage

        pct = compute_cloud_coverage(scl_array_clean)
        assert pct == pytest.approx(0.0)

    def test_compute_cloud_coverage_half(self, scl_array_half_cloud):
        from roofscan.core.preproceso.cloud_mask import compute_cloud_coverage

        pct = compute_cloud_coverage(scl_array_half_cloud)
        assert pct == pytest.approx(50.0, abs=1.0)

    def test_scl_class_summary_returns_dict(self, scl_array_half_cloud):
        from roofscan.core.preproceso.cloud_mask import scl_class_summary

        summary = scl_class_summary(scl_array_half_cloud)
        assert isinstance(summary, dict)
        assert len(summary) > 0
        assert all(isinstance(v, float) for v in summary.values())

    def test_scl_class_summary_percentages_sum_100(self, scl_array_half_cloud):
        from roofscan.core.preproceso.cloud_mask import scl_class_summary

        summary = scl_class_summary(scl_array_half_cloud)
        total = sum(summary.values())
        assert total == pytest.approx(100.0, abs=0.1)


# ---------------------------------------------------------------------------
# Tests: normalizer
# ---------------------------------------------------------------------------

class TestNormalizer:
    def test_normalize_s2_range_0_1(self, spectral_array_utm):
        from roofscan.core.preproceso.normalizer import normalize_s2

        normed = normalize_s2(spectral_array_utm.astype(np.float32))
        valid = normed[~np.isnan(normed)]
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    def test_normalize_s2_output_dtype_float32(self, spectral_array_utm):
        from roofscan.core.preproceso.normalizer import normalize_s2

        normed = normalize_s2(spectral_array_utm)
        assert normed.dtype == np.float32

    def test_normalize_s2_nodata_becomes_nan(self):
        from roofscan.core.preproceso.normalizer import normalize_s2

        arr = np.array([[[0, 500, 1000]]], dtype=np.uint16)  # 0 = nodata
        normed = normalize_s2(arr, nodata=0)
        assert np.isnan(normed[0, 0, 0])
        assert not np.isnan(normed[0, 0, 1])

    def test_normalize_s2_known_value(self):
        from roofscan.core.preproceso.normalizer import normalize_s2

        arr = np.array([[[5000]]], dtype=np.uint16)
        normed = normalize_s2(arr)
        assert normed[0, 0, 0] == pytest.approx(0.5, abs=1e-4)

    def test_clip_percentile_reduces_range(self):
        from roofscan.core.preproceso.normalizer import clip_percentile

        arr = np.arange(100, dtype=np.float32).reshape(1, 10, 10)
        clipped = clip_percentile(arr, low_pct=10, high_pct=90)
        assert clipped.min() >= arr.min()
        assert clipped.max() <= arr.max()

    def test_clip_percentile_invalid_args_raises(self):
        from roofscan.core.preproceso.normalizer import clip_percentile

        with pytest.raises(ValueError, match="Percentiles"):
            clip_percentile(np.ones((1, 5, 5)), low_pct=90, high_pct=10)

    def test_band_statistics_returns_one_dict_per_band(self):
        from roofscan.core.preproceso.normalizer import band_statistics

        arr = np.random.rand(4, 20, 20).astype(np.float32)
        stats = band_statistics(arr)
        assert len(stats) == 4
        for s in stats:
            assert "min" in s and "max" in s and "mean" in s

    def test_band_statistics_all_nan_band(self):
        from roofscan.core.preproceso.normalizer import band_statistics

        arr = np.full((2, 10, 10), np.nan, dtype=np.float32)
        stats = band_statistics(arr)
        assert stats[0]["valid_pct"] == 0.0


# ---------------------------------------------------------------------------
# Tests: pipeline
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_pipeline_full_run(self, data_utm, scl_array_clean):
        from roofscan.core.preproceso.pipeline import run_preprocessing

        result = run_preprocessing(data_utm, scl_array=scl_array_clean)
        assert "array" in result
        assert "valid_mask" in result
        assert "cloud_pct" in result
        assert "stats" in result

    def test_pipeline_array_is_float32(self, data_utm, scl_array_clean):
        from roofscan.core.preproceso.pipeline import run_preprocessing

        result = run_preprocessing(data_utm, scl_array=scl_array_clean)
        assert result["array"].dtype == np.float32

    def test_pipeline_normalized_range(self, data_utm, scl_array_clean):
        from roofscan.core.preproceso.pipeline import run_preprocessing

        result = run_preprocessing(data_utm, scl_array=scl_array_clean)
        valid = result["array"][~np.isnan(result["array"])]
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    def test_pipeline_no_scl_skips_cloud_mask(self, data_utm):
        from roofscan.core.preproceso.pipeline import run_preprocessing

        result = run_preprocessing(data_utm, scl_array=None)
        assert result["valid_mask"] is None
        assert result["cloud_pct"] is None

    def test_pipeline_skip_reproject(self, data_utm, scl_array_clean):
        from roofscan.core.preproceso.pipeline import run_preprocessing, PreprocessConfig

        config = PreprocessConfig(skip_reproject=True)
        result = run_preprocessing(data_utm, scl_array=scl_array_clean, config=config)
        assert result["crs"] == data_utm["crs"]

    def test_pipeline_cloud_pct_with_half_cloud(self, data_utm, scl_array_half_cloud):
        from roofscan.core.preproceso.pipeline import run_preprocessing

        result = run_preprocessing(data_utm, scl_array=scl_array_half_cloud)
        assert result["cloud_pct"] == pytest.approx(50.0, abs=1.0)

    def test_pipeline_does_not_mutate_input(self, data_utm, scl_array_clean):
        """El pipeline no debe modificar el dict original."""
        from roofscan.core.preproceso.pipeline import run_preprocessing

        original_array = data_utm["array"].copy()
        run_preprocessing(data_utm, scl_array=scl_array_clean)
        np.testing.assert_array_equal(data_utm["array"], original_array)


# ---------------------------------------------------------------------------
# Tests: raster_exporter
# ---------------------------------------------------------------------------

class TestRasterExporter:
    def test_export_geotiff_creates_file(self, data_utm, tmp_path):
        from roofscan.core.exportacion.raster_exporter import export_geotiff

        path = export_geotiff(data_utm, tmp_path, filename="test_out")
        assert path.exists()
        assert path.suffix == ".tif"

    def test_export_geotiff_is_valid_raster(self, data_utm, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        from roofscan.core.exportacion.raster_exporter import export_geotiff

        path = export_geotiff(data_utm, tmp_path, filename="test_valid")
        with rasterio.open(path) as src:
            assert src.count == data_utm["array"].shape[0]
            assert "32720" in src.crs.to_string()

    def test_export_preview_png_creates_file(self, data_utm, tmp_path):
        pytest.importorskip("matplotlib")
        from roofscan.core.exportacion.raster_exporter import export_preview_png

        # Normalizar el array para que esté en [0,1]
        data = dict(data_utm)
        data["array"] = data["array"].astype(np.float32) / 10000.0

        path = export_preview_png(data, tmp_path, filename="test_preview")
        assert path.exists()
        assert path.suffix == ".png"

    def test_export_preview_invalid_band_raises(self, data_utm, tmp_path):
        pytest.importorskip("matplotlib")
        from roofscan.core.exportacion.raster_exporter import export_preview_png

        data = dict(data_utm)
        data["array"] = data["array"].astype(np.float32) / 10000.0

        with pytest.raises(ValueError, match="banda"):
            export_preview_png(data, tmp_path, rgb_bands=(99, 2, 1))

    def test_export_geotiff_missing_key_raises(self, tmp_path):
        from roofscan.core.exportacion.raster_exporter import export_geotiff

        with pytest.raises(ValueError, match="claves requeridas"):
            export_geotiff({"array": np.zeros((1, 5, 5))}, tmp_path)
