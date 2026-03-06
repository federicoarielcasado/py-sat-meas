"""Tests del módulo de ingesta (downloader + loader).

Ejecutar sin credenciales de red:
    pytest tests/test_ingesta.py -m "not network"

Ejecutar con credenciales CDSE reales (requiere .env):
    pytest tests/test_ingesta.py -m network
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_tif_epsg32720(tmp_path: Path) -> Path:
    """GeoTIFF sintético válido en EPSG:32720 (UTM 20S), 2 bandas, 10 m/px."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    width, height = 50, 50
    # Extent en UTM 20S aproximado a Luján
    west, south, east, north = 343000, 6172000, 343500, 6172500
    transform = from_bounds(west, south, east, north, width, height)
    crs = CRS.from_epsg(32720)

    filepath = tmp_path / "test_image.tif"
    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=2,
        dtype="uint16",
        crs=crs,
        transform=transform,
        nodata=0,
    ) as dst:
        data = np.random.randint(100, 3000, (2, height, width), dtype=np.uint16)
        dst.write(data)

    return filepath


@pytest.fixture()
def tmp_tif_no_crs(tmp_path: Path) -> Path:
    """GeoTIFF sintético sin CRS definido."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    width, height = 20, 20
    transform = from_bounds(0, 0, 1, 1, width, height)

    filepath = tmp_path / "no_crs.tif"
    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        transform=transform,
    ) as dst:
        dst.write(np.zeros((1, height, width), dtype=np.uint8))

    return filepath


# ---------------------------------------------------------------------------
# Tests: loader.load_geotiff
# ---------------------------------------------------------------------------

class TestLoadGeotiff:
    def test_valid_file_returns_expected_keys(self, tmp_tif_epsg32720: Path):
        from roofscan.core.ingesta.loader import load_geotiff

        data = load_geotiff(tmp_tif_epsg32720)

        expected_keys = {"array", "crs", "transform", "bounds", "resolution_m", "nodata", "count", "dtype", "sensor", "filepath"}
        assert expected_keys == set(data.keys())

    def test_valid_file_array_shape(self, tmp_tif_epsg32720: Path):
        from roofscan.core.ingesta.loader import load_geotiff

        data = load_geotiff(tmp_tif_epsg32720)
        assert data["array"].ndim == 3  # (bandas, alto, ancho)
        assert data["count"] == 2
        assert data["array"].shape[0] == 2

    def test_valid_file_crs_detected(self, tmp_tif_epsg32720: Path):
        from roofscan.core.ingesta.loader import load_geotiff

        data = load_geotiff(tmp_tif_epsg32720)
        assert "32720" in data["crs"]

    def test_valid_file_resolution_10m(self, tmp_tif_epsg32720: Path):
        from roofscan.core.ingesta.loader import load_geotiff

        data = load_geotiff(tmp_tif_epsg32720)
        # extent 500 m / 50 px = 10 m/px
        assert data["resolution_m"] == pytest.approx(10.0, abs=0.1)

    def test_valid_file_sensor_detected_as_sentinel2(self, tmp_tif_epsg32720: Path):
        from roofscan.core.ingesta.loader import load_geotiff

        data = load_geotiff(tmp_tif_epsg32720)
        assert data["sensor"] == "Sentinel-2"

    def test_valid_file_filepath_is_absolute(self, tmp_tif_epsg32720: Path):
        from roofscan.core.ingesta.loader import load_geotiff

        data = load_geotiff(tmp_tif_epsg32720)
        assert Path(data["filepath"]).is_absolute()

    def test_missing_file_raises_file_not_found(self, tmp_path: Path):
        from roofscan.core.ingesta.loader import load_geotiff

        with pytest.raises(FileNotFoundError, match="No se encontró"):
            load_geotiff(tmp_path / "inexistente.tif")

    def test_no_crs_raises_value_error(self, tmp_tif_no_crs: Path):
        from roofscan.core.ingesta.loader import load_geotiff

        with pytest.raises(ValueError, match="CRS"):
            load_geotiff(tmp_tif_no_crs)

    def test_invalid_path_type_string_accepted(self, tmp_tif_epsg32720: Path):
        from roofscan.core.ingesta.loader import load_geotiff

        # Debe aceptar strings, no solo Path
        data = load_geotiff(str(tmp_tif_epsg32720))
        assert "array" in data


# ---------------------------------------------------------------------------
# Tests: loader.detect_sensor
# ---------------------------------------------------------------------------

class TestDetectSensor:
    @pytest.mark.parametrize("resolution_m,expected", [
        (10.0, "Sentinel-2"),
        (9.8, "Sentinel-2"),    # dentro de tolerancia
        (30.0, "Landsat-8/9"),
        (1.5, "SPOT-6/7"),
        (0.5, "Pleiades"),
        (5.0, "desconocido"),
        (None, "desconocido"),
    ])
    def test_sensor_detection(self, resolution_m, expected):
        from roofscan.core.ingesta.loader import detect_sensor

        result = detect_sensor({"resolution_m": resolution_m, "count": 4})
        assert result == expected


# ---------------------------------------------------------------------------
# Tests: downloader — validaciones sin red
# ---------------------------------------------------------------------------

class TestDownloaderValidations:
    def test_invalid_bbox_wrong_length(self):
        from roofscan.core.ingesta.downloader import search_sentinel2

        with pytest.raises(ValueError, match="bbox"):
            search_sentinel2((1.0, 2.0), ("2024-01-01", "2024-03-31"))

    def test_invalid_bbox_lon_order(self):
        from roofscan.core.ingesta.downloader import search_sentinel2

        with pytest.raises(ValueError, match="lon_min"):
            search_sentinel2((-59.05, -34.60, -59.15, -34.53), ("2024-01-01", "2024-03-31"))

    def test_invalid_bbox_lat_order(self):
        from roofscan.core.ingesta.downloader import search_sentinel2

        with pytest.raises(ValueError, match="lat_min"):
            search_sentinel2((-59.15, -34.53, -59.05, -34.60), ("2024-01-01", "2024-03-31"))

    def test_invalid_date_format(self):
        from roofscan.core.ingesta.downloader import search_sentinel2

        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            search_sentinel2((-59.15, -34.60, -59.05, -34.53), ("01/01/2024", "31/03/2024"))

    def test_inverted_date_range(self):
        from roofscan.core.ingesta.downloader import search_sentinel2

        with pytest.raises(ValueError, match="anterior"):
            search_sentinel2((-59.15, -34.60, -59.05, -34.53), ("2024-03-31", "2024-01-01"))

    def test_missing_credentials_raises_env_error(self, monkeypatch):
        # search_sentinel2 usa la STAC API pública de CDSE (sin autenticación).
        # Las credenciales solo se validan en download_sentinel2_scene / _get_token.
        # Este test verifica que _check_credentials() lanza EnvironmentError.
        from roofscan.core.ingesta.downloader import _check_credentials

        monkeypatch.delenv("CDSE_USER", raising=False)
        monkeypatch.delenv("CDSE_PASSWORD", raising=False)

        with pytest.raises(EnvironmentError, match="Credenciales"):
            _check_credentials()


# ---------------------------------------------------------------------------
# Tests de red (requieren credenciales reales en .env)
# ---------------------------------------------------------------------------

@pytest.mark.network
class TestDownloaderNetwork:
    """Tests que requieren conexión a internet y credenciales CDSE.

    Ejecutar explícitamente con:
        pytest tests/test_ingesta.py -m network
    """

    def test_search_returns_results_for_lujan(self):
        """Verifica que la búsqueda retorna al menos una escena para Luján."""
        from roofscan.config import LUJAN_BBOX_WGS84
        from roofscan.core.ingesta.downloader import search_sentinel2

        results = search_sentinel2(
            bbox=LUJAN_BBOX_WGS84,
            date_range=("2024-01-01", "2024-06-30"),
            max_cloud_pct=30,
            max_results=5,
        )

        assert len(results) > 0, "Se esperaba al menos una escena para Luján en el período indicado"

    def test_search_result_has_required_fields(self):
        from roofscan.config import LUJAN_BBOX_WGS84
        from roofscan.core.ingesta.downloader import search_sentinel2

        results = search_sentinel2(
            bbox=LUJAN_BBOX_WGS84,
            date_range=("2024-01-01", "2024-06-30"),
            max_cloud_pct=30,
            max_results=3,
        )

        if results:
            scene = results[0]
            for field in ("id", "name", "date", "cloud_pct", "size_mb"):
                assert field in scene, f"Campo '{field}' ausente en resultado de búsqueda"

    def test_search_cloud_pct_filter_respected(self):
        from roofscan.config import LUJAN_BBOX_WGS84
        from roofscan.core.ingesta.downloader import search_sentinel2

        max_cloud = 15.0
        results = search_sentinel2(
            bbox=LUJAN_BBOX_WGS84,
            date_range=("2024-01-01", "2024-12-31"),
            max_cloud_pct=max_cloud,
            max_results=5,
        )

        for scene in results:
            assert scene["cloud_pct"] <= max_cloud, (
                f"Escena {scene['name']} tiene {scene['cloud_pct']}% de nubosidad, "
                f"superando el límite de {max_cloud}%"
            )
