"""Tests para los módulos de cálculo y exportación.

Cubre:
- roofscan.core.calculo.classifier
- roofscan.core.exportacion.geojson_exporter
- roofscan.core.exportacion.shp_exporter

Todos los tests son unitarios: sin red, sin GPU, sin archivos externos.
"""

import math
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_square_polygon(side_m: float = 100.0):
    """Polígono cuadrado de lado ``side_m`` m (en coords proyectadas)."""
    from shapely.geometry import Polygon
    return Polygon([(0, 0), (side_m, 0), (side_m, side_m), (0, side_m)])


def _make_elongated_polygon(long_m: float = 500.0, short_m: float = 30.0):
    """Polígono muy elongado (galpón)."""
    from shapely.geometry import Polygon
    return Polygon([(0, 0), (long_m, 0), (long_m, short_m), (0, short_m)])


def _make_roof_gdf(rows: list[dict]) -> "gpd.GeoDataFrame":
    """Construye un GeoDataFrame mínimo con columnas area_m2 y geometry."""
    gpd = pytest.importorskip("geopandas")
    return gpd.GeoDataFrame(rows, crs="EPSG:32720")


# ---------------------------------------------------------------------------
# Tests: compute_shape_metrics
# ---------------------------------------------------------------------------

class TestComputeShapeMetrics:
    def test_square_compactness_close_to_pi_over_4(self):
        from roofscan.core.calculo.classifier import compute_shape_metrics
        poly = _make_square_polygon(100)
        m = compute_shape_metrics(poly)
        # Compacidad de cuadrado: π/4 ≈ 0.785
        assert abs(m["compactness"] - math.pi / 4) < 0.01

    def test_elongated_low_compactness(self):
        from roofscan.core.calculo.classifier import compute_shape_metrics
        poly = _make_elongated_polygon(500, 30)
        m = compute_shape_metrics(poly)
        assert m["compactness"] < 0.45

    def test_elongated_high_elongation(self):
        from roofscan.core.calculo.classifier import compute_shape_metrics
        poly = _make_elongated_polygon(500, 30)
        m = compute_shape_metrics(poly)
        assert m["elongation"] > 2.0

    def test_square_elongation_close_to_one(self):
        from roofscan.core.calculo.classifier import compute_shape_metrics
        poly = _make_square_polygon(100)
        m = compute_shape_metrics(poly)
        assert m["elongation"] < 1.5

    def test_area_and_perimeter_match(self):
        from roofscan.core.calculo.classifier import compute_shape_metrics
        poly = _make_square_polygon(50)
        m = compute_shape_metrics(poly)
        assert abs(m["area"] - 50 ** 2) < 0.1
        assert abs(m["perimeter"] - 4 * 50) < 0.1


# ---------------------------------------------------------------------------
# Tests: classify_by_geometry
# ---------------------------------------------------------------------------

class TestClassifyByGeometry:
    def _gdf(self, rows):
        return _make_roof_gdf(rows)

    def test_small_compact_is_vivienda(self):
        from roofscan.core.calculo.classifier import classify_by_geometry, LABEL_VIVIENDA
        gdf = self._gdf([{"area_m2": 120.0, "geometry": _make_square_polygon(11)}])
        result = classify_by_geometry(gdf)
        assert result["tipo_estructura"].iloc[0] == LABEL_VIVIENDA

    def test_large_elongated_is_galpon(self):
        from roofscan.core.calculo.classifier import classify_by_geometry, LABEL_GALPON
        gdf = self._gdf([{"area_m2": 450.0, "geometry": _make_elongated_polygon(500, 30)}])
        result = classify_by_geometry(gdf)
        assert result["tipo_estructura"].iloc[0] == LABEL_GALPON

    def test_very_large_is_industrial(self):
        from roofscan.core.calculo.classifier import classify_by_geometry, LABEL_INDUSTRIAL
        gdf = self._gdf([{"area_m2": 2000.0, "geometry": _make_square_polygon(45)}])
        result = classify_by_geometry(gdf)
        assert result["tipo_estructura"].iloc[0] == LABEL_INDUSTRIAL

    def test_output_col_added(self):
        from roofscan.core.calculo.classifier import classify_by_geometry
        gdf = self._gdf([{"area_m2": 100.0, "geometry": _make_square_polygon(10)}])
        result = classify_by_geometry(gdf)
        assert "tipo_estructura" in result.columns

    def test_custom_out_col(self):
        from roofscan.core.calculo.classifier import classify_by_geometry
        gdf = self._gdf([{"area_m2": 100.0, "geometry": _make_square_polygon(10)}])
        result = classify_by_geometry(gdf, out_col="categoria")
        assert "categoria" in result.columns

    def test_missing_area_col_raises(self):
        from roofscan.core.calculo.classifier import classify_by_geometry
        gpd = pytest.importorskip("geopandas")
        gdf = gpd.GeoDataFrame([{"geometry": _make_square_polygon()}], crs="EPSG:32720")
        with pytest.raises(KeyError):
            classify_by_geometry(gdf)

    def test_empty_geometry_returns_otro(self):
        from roofscan.core.calculo.classifier import classify_by_geometry, LABEL_OTRO
        pytest.importorskip("geopandas")
        from shapely.geometry import Polygon
        gdf = self._gdf([{"area_m2": 100.0, "geometry": Polygon()}])
        result = classify_by_geometry(gdf)
        assert result["tipo_estructura"].iloc[0] == LABEL_OTRO

    def test_does_not_mutate_input(self):
        from roofscan.core.calculo.classifier import classify_by_geometry
        gdf = self._gdf([{"area_m2": 100.0, "geometry": _make_square_polygon()}])
        original_cols = list(gdf.columns)
        classify_by_geometry(gdf)
        assert list(gdf.columns) == original_cols

    def test_multiple_rows(self):
        from roofscan.core.calculo.classifier import classify_by_geometry
        gdf = self._gdf([
            {"area_m2": 100.0, "geometry": _make_square_polygon(10)},
            {"area_m2": 2500.0, "geometry": _make_elongated_polygon(1000, 100)},
        ])
        result = classify_by_geometry(gdf)
        assert len(result) == 2
        assert result["tipo_estructura"].notna().all()


# ---------------------------------------------------------------------------
# Tests: classify_parcela
# ---------------------------------------------------------------------------

class TestClassifyParcela:
    def _gdf(self, rows):
        gpd = pytest.importorskip("geopandas")
        return gpd.GeoDataFrame(rows, crs="EPSG:4326")

    def test_large_area_is_industrial(self):
        from roofscan.core.calculo.classifier import classify_parcela, LABEL_INDUSTRIAL
        gdf = self._gdf([{"area_techos_m2": 2000.0, "pct_cubierto": 40.0}])
        result = classify_parcela(gdf)
        assert result["tipo_predominante"].iloc[0] == LABEL_INDUSTRIAL

    def test_medium_low_pct_is_galpon(self):
        from roofscan.core.calculo.classifier import classify_parcela, LABEL_GALPON
        gdf = self._gdf([{"area_techos_m2": 600.0, "pct_cubierto": 30.0}])
        result = classify_parcela(gdf)
        assert result["tipo_predominante"].iloc[0] == LABEL_GALPON

    def test_small_is_vivienda(self):
        from roofscan.core.calculo.classifier import classify_parcela, LABEL_VIVIENDA
        gdf = self._gdf([{"area_techos_m2": 150.0, "pct_cubierto": 55.0}])
        result = classify_parcela(gdf)
        assert result["tipo_predominante"].iloc[0] == LABEL_VIVIENDA

    def test_zero_area_is_otro(self):
        from roofscan.core.calculo.classifier import classify_parcela, LABEL_OTRO
        gdf = self._gdf([{"area_techos_m2": 0.0, "pct_cubierto": 0.0}])
        result = classify_parcela(gdf)
        assert result["tipo_predominante"].iloc[0] == LABEL_OTRO

    def test_missing_col_raises(self):
        from roofscan.core.calculo.classifier import classify_parcela
        gpd = pytest.importorskip("geopandas")
        gdf = gpd.GeoDataFrame([{"area_techos_m2": 100.0}], crs="EPSG:4326")
        with pytest.raises(KeyError):
            classify_parcela(gdf)

    def test_custom_out_col(self):
        from roofscan.core.calculo.classifier import classify_parcela
        gdf = self._gdf([{"area_techos_m2": 150.0, "pct_cubierto": 50.0}])
        result = classify_parcela(gdf, out_col="tipo")
        assert "tipo" in result.columns


# ---------------------------------------------------------------------------
# Tests: geojson_exporter
# ---------------------------------------------------------------------------

class TestExportGeojson:
    def _gdf(self, tmp_path):
        gpd = pytest.importorskip("geopandas")
        return gpd.GeoDataFrame(
            [{"area_m2": 100.0, "geometry": _make_square_polygon()}],
            crs="EPSG:32720",
        )

    def test_creates_file(self, tmp_path):
        pytest.importorskip("geopandas")
        from roofscan.core.exportacion.geojson_exporter import export_geojson
        gdf = self._gdf(tmp_path)
        out = export_geojson(gdf, tmp_path / "test.geojson")
        assert out.exists()

    def test_returns_path(self, tmp_path):
        pytest.importorskip("geopandas")
        from roofscan.core.exportacion.geojson_exporter import export_geojson
        gdf = self._gdf(tmp_path)
        out = export_geojson(gdf, tmp_path / "test.geojson")
        assert isinstance(out, Path)

    def test_creates_parent_dir(self, tmp_path):
        pytest.importorskip("geopandas")
        from roofscan.core.exportacion.geojson_exporter import export_geojson
        gdf = self._gdf(tmp_path)
        out = export_geojson(gdf, tmp_path / "subdir" / "test.geojson")
        assert out.exists()

    def test_empty_gdf_raises(self, tmp_path):
        gpd = pytest.importorskip("geopandas")
        from roofscan.core.exportacion.geojson_exporter import export_geojson
        gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:32720")
        with pytest.raises(ValueError, match="vacío"):
            export_geojson(gdf, tmp_path / "test.geojson")

    def test_readable_back(self, tmp_path):
        gpd = pytest.importorskip("geopandas")
        from roofscan.core.exportacion.geojson_exporter import export_geojson
        gdf = self._gdf(tmp_path)
        out = export_geojson(gdf, tmp_path / "test.geojson")
        loaded = gpd.read_file(out)
        assert len(loaded) == 1
        assert "area_m2" in loaded.columns


# ---------------------------------------------------------------------------
# Tests: shp_exporter
# ---------------------------------------------------------------------------

class TestExportShapefile:
    def _gdf(self):
        gpd = pytest.importorskip("geopandas")
        return gpd.GeoDataFrame(
            [{"area_m2": 100.0, "centroid_x_m": 1.0, "geometry": _make_square_polygon()}],
            crs="EPSG:32720",
        )

    def test_creates_shp(self, tmp_path):
        pytest.importorskip("geopandas")
        from roofscan.core.exportacion.shp_exporter import export_shapefile
        out = export_shapefile(self._gdf(), tmp_path / "test.shp")
        assert out.exists()

    def test_returns_path(self, tmp_path):
        pytest.importorskip("geopandas")
        from roofscan.core.exportacion.shp_exporter import export_shapefile
        out = export_shapefile(self._gdf(), tmp_path / "test.shp")
        assert isinstance(out, Path)

    def test_creates_sidecar_files(self, tmp_path):
        pytest.importorskip("geopandas")
        from roofscan.core.exportacion.shp_exporter import export_shapefile
        export_shapefile(self._gdf(), tmp_path / "test.shp")
        assert (tmp_path / "test.dbf").exists()
        assert (tmp_path / "test.shx").exists()

    def test_renames_long_columns(self, tmp_path):
        gpd = pytest.importorskip("geopandas")
        from roofscan.core.exportacion.shp_exporter import export_shapefile
        # centroid_x_m (11 chars) → cent_x_m (8 chars)
        out = export_shapefile(self._gdf(), tmp_path / "test.shp")
        loaded = gpd.read_file(out)
        assert "cent_x_m" in loaded.columns
        assert "centroid_x_m" not in loaded.columns

    def test_empty_gdf_raises(self, tmp_path):
        gpd = pytest.importorskip("geopandas")
        from roofscan.core.exportacion.shp_exporter import export_shapefile
        gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:32720")
        with pytest.raises(ValueError, match="vacío"):
            export_shapefile(gdf, tmp_path / "test.shp")

    def test_creates_parent_dir(self, tmp_path):
        pytest.importorskip("geopandas")
        from roofscan.core.exportacion.shp_exporter import export_shapefile
        out = export_shapefile(self._gdf(), tmp_path / "subdir" / "test.shp")
        assert out.exists()
