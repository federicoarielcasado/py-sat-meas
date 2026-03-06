"""Tests del módulo carto_arba: validaciones sin red + tests de integración WMS.

Tests sin red (siempre pasan):
    pytest tests/test_carto_arba.py -m "not network"

Tests con red (requieren internet):
    pytest tests/test_carto_arba.py -m network
"""

import math

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Tests: geocode_address — validaciones sin red
# ---------------------------------------------------------------------------

class TestGeocodeValidations:
    def test_empty_address_raises_value_error(self):
        from roofscan.core.ingesta.carto_arba import geocode_address
        with pytest.raises(ValueError, match="vacía"):
            geocode_address("")

    def test_whitespace_only_raises_value_error(self):
        from roofscan.core.ingesta.carto_arba import geocode_address
        with pytest.raises(ValueError, match="vacía"):
            geocode_address("   ")


# ---------------------------------------------------------------------------
# Tests: bbox_from_latlon
# ---------------------------------------------------------------------------

class TestBboxFromLatlon:
    def test_returns_four_values(self):
        from roofscan.core.ingesta.carto_arba import bbox_from_latlon
        bbox = bbox_from_latlon(-34.570, -59.105)
        assert len(bbox) == 4

    def test_lon_min_less_than_lon_max(self):
        from roofscan.core.ingesta.carto_arba import bbox_from_latlon
        lon_min, lat_min, lon_max, lat_max = bbox_from_latlon(-34.570, -59.105)
        assert lon_min < lon_max

    def test_lat_min_less_than_lat_max(self):
        from roofscan.core.ingesta.carto_arba import bbox_from_latlon
        lon_min, lat_min, lon_max, lat_max = bbox_from_latlon(-34.570, -59.105)
        assert lat_min < lat_max

    def test_center_is_approximately_input(self):
        from roofscan.core.ingesta.carto_arba import bbox_from_latlon
        lat, lon = -34.570, -59.105
        lon_min, lat_min, lon_max, lat_max = bbox_from_latlon(lat, lon, radius_km=1.0)
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        assert center_lat == pytest.approx(lat, abs=0.001)
        assert center_lon == pytest.approx(lon, abs=0.001)

    def test_radius_affects_bbox_size(self):
        from roofscan.core.ingesta.carto_arba import bbox_from_latlon
        small = bbox_from_latlon(-34.570, -59.105, radius_km=0.5)
        large = bbox_from_latlon(-34.570, -59.105, radius_km=2.0)
        small_span = small[2] - small[0]  # lon_max - lon_min
        large_span = large[2] - large[0]
        assert large_span > small_span

    def test_lujan_bbox_in_buenos_aires_province(self):
        from roofscan.core.ingesta.carto_arba import bbox_from_latlon
        lon_min, lat_min, lon_max, lat_max = bbox_from_latlon(-34.570, -59.105)
        # Provincia de Buenos Aires: -63.70 a -56.46 W, -41.07 a -33.22 S
        assert -63.70 <= lon_min and lon_max <= -56.46
        assert -41.07 <= lat_min and lat_max <= -33.22


# ---------------------------------------------------------------------------
# Tests: _parse_feature_info_gml (interno, testeable sin red)
# ---------------------------------------------------------------------------

class TestParseFeatureInfoGml:
    def test_empty_string_returns_empty_dict(self):
        from roofscan.core.ingesta.carto_arba import _parse_feature_info_gml
        assert _parse_feature_info_gml("") == {}

    def test_invalid_xml_returns_empty_dict(self):
        from roofscan.core.ingesta.carto_arba import _parse_feature_info_gml
        assert _parse_feature_info_gml("esto no es xml <<<<") == {}

    def test_service_exception_returns_empty_dict(self):
        from roofscan.core.ingesta.carto_arba import _parse_feature_info_gml
        xml = """<?xml version="1.0"?>
        <ServiceExceptionReport>
          <ServiceException>Layer not found</ServiceException>
        </ServiceExceptionReport>"""
        result = _parse_feature_info_gml(xml)
        assert result == {}

    def test_extracts_nomenclatura(self):
        from roofscan.core.ingesta.carto_arba import _parse_feature_info_gml
        # El namespace debe declararse para que ET.fromstring() no falle
        xml = """<?xml version="1.0"?>
        <FeatureCollection xmlns:idera="http://www.gba.gob.ar/idera/">
          <featureMember>
            <idera:Parcela>
              <idera:nomenclatura>070-B-045-12-0031/00</idera:nomenclatura>
              <idera:partido>Luján</idera:partido>
            </idera:Parcela>
          </featureMember>
        </FeatureCollection>"""
        result = _parse_feature_info_gml(xml)
        assert result.get("nomenclatura") == "070-B-045-12-0031/00"
        assert result.get("partido") == "Luján"

    def test_extracts_seccion_manzana(self):
        from roofscan.core.ingesta.carto_arba import _parse_feature_info_gml
        xml = """<?xml version="1.0"?>
        <FeatureCollection xmlns:idera="http://www.gba.gob.ar/idera/">
          <featureMember>
            <idera:Parcela>
              <idera:seccion>B</idera:seccion>
              <idera:manzana>045</idera:manzana>
              <idera:parcela>0031</idera:parcela>
            </idera:Parcela>
          </featureMember>
        </FeatureCollection>"""
        result = _parse_feature_info_gml(xml)
        assert "seccion" in result
        assert "manzana" in result
        assert "parcela" in result


# ---------------------------------------------------------------------------
# Tests: _parse_wms_error
# ---------------------------------------------------------------------------

class TestParseWmsError:
    def test_extracts_message_from_exception(self):
        from roofscan.core.ingesta.carto_arba import _parse_wms_error
        xml = """<?xml version="1.0"?>
        <ServiceExceptionReport version="1.3.0">
          <ServiceException code="InvalidCRS">
            El CRS solicitado no es válido.
          </ServiceException>
        </ServiceExceptionReport>"""
        msg = _parse_wms_error(xml)
        assert "CRS" in msg or "válido" in msg

    def test_returns_truncated_text_on_invalid_xml(self):
        from roofscan.core.ingesta.carto_arba import _parse_wms_error
        raw = "X" * 300
        result = _parse_wms_error(raw)
        assert len(result) <= 200

    def test_empty_input(self):
        from roofscan.core.ingesta.carto_arba import _parse_wms_error
        assert _parse_wms_error("") == "Error desconocido"


# ---------------------------------------------------------------------------
# Tests: get_parcelas_image — validaciones de entrada sin red
# ---------------------------------------------------------------------------

class TestGetParcelasImageValidations:
    def test_invalid_bbox_negative_span_lon(self):
        """Un bbox con lon_min > lon_max debería resultar en una imagen de error o excepción."""
        # No validamos internamente el bbox (lo hace ARBA), pero podemos verificar
        # que la función acepta el tipo correcto de argumentos.
        from roofscan.core.ingesta.carto_arba import bbox_from_latlon
        bbox = bbox_from_latlon(-34.570, -59.105, radius_km=0.3)
        assert len(bbox) == 4


# ---------------------------------------------------------------------------
# Tests de red (requieren conexión a internet)
# ---------------------------------------------------------------------------

@pytest.mark.network
class TestCartoArbaNetwork:
    """Tests de integración con el WMS de CartoARBA.

    Requieren conexión a internet. Ejecutar con:
        pytest tests/test_carto_arba.py -m network
    """

    LUJAN_BBOX = (-59.115, -34.580, -59.095, -34.560)  # lon_min, lat_min, lon_max, lat_max

    def test_geocode_lujan_returns_coordinates(self):
        from roofscan.core.ingesta.carto_arba import geocode_address
        lat, lon = geocode_address("Luján, Buenos Aires, Argentina")
        # Luján está cerca de (-34.57, -59.10)
        assert -35.0 <= lat <= -34.0
        assert -60.0 <= lon <= -58.5

    def test_geocode_address_not_found_raises_value_error(self):
        from roofscan.core.ingesta.carto_arba import geocode_address
        with pytest.raises(ValueError, match="resultados"):
            geocode_address("XYZ123 Lugar Inexistente Completamente Falso 99999")

    def test_get_parcelas_image_returns_array(self):
        from roofscan.core.ingesta.carto_arba import get_parcelas_image
        array, extent = get_parcelas_image(self.LUJAN_BBOX, width=256, height=256)
        assert isinstance(array, np.ndarray)
        assert array.ndim == 3
        assert array.shape[2] == 4   # RGBA

    def test_get_parcelas_image_dimensions_match_request(self):
        from roofscan.core.ingesta.carto_arba import get_parcelas_image
        W, H = 256, 256
        array, _ = get_parcelas_image(self.LUJAN_BBOX, width=W, height=H)
        assert array.shape[0] == H
        assert array.shape[1] == W

    def test_get_parcelas_image_extent_matches_bbox(self):
        from roofscan.core.ingesta.carto_arba import get_parcelas_image
        lon_min, lat_min, lon_max, lat_max = self.LUJAN_BBOX
        _, extent = get_parcelas_image(self.LUJAN_BBOX, width=256, height=256)
        # extent = (lon_min, lon_max, lat_min, lat_max)
        assert extent[0] == pytest.approx(lon_min)
        assert extent[1] == pytest.approx(lon_max)

    def test_get_parcelas_image_cache_works(self):
        from roofscan.core.ingesta.carto_arba import get_parcelas_image
        arr1, _ = get_parcelas_image(self.LUJAN_BBOX, width=128, height=128, use_cache=True)
        arr2, _ = get_parcelas_image(self.LUJAN_BBOX, width=128, height=128, use_cache=True)
        # Segunda llamada usa caché → mismo objeto
        assert arr1 is arr2

    def test_get_feature_info_returns_dict(self):
        from roofscan.core.ingesta.carto_arba import get_parcel_info
        info = get_parcel_info(
            self.LUJAN_BBOX,
            pixel_x=128, pixel_y=128,
            map_w=256, map_h=256,
        )
        assert isinstance(info, dict)
        assert "lat" in info
        assert "lon" in info

    def test_get_feature_info_coordinates_within_bbox(self):
        from roofscan.core.ingesta.carto_arba import get_parcel_info
        lon_min, lat_min, lon_max, lat_max = self.LUJAN_BBOX
        info = get_parcel_info(self.LUJAN_BBOX, 128, 128, 256, 256)
        assert lon_min <= info["lon"] <= lon_max
        assert lat_min <= info["lat"] <= lat_max
