"""Cliente WMS para CartoARBA (ARBA — Provincia de Buenos Aires).

Accede a la capa catastral de parcelas de ARBA a través de su servicio
WMS público (IDERA). No requiere autenticación ni credenciales.

Servicios disponibles:
    WMS: https://geo.arba.gov.ar/geoserver/idera/wms  (activo)
    WFS: deshabilitado en el servidor actual

Capas principales:
    idera:Parcela  — parcelas catastrales individuales
    idera:Manzana  — manzanas urbanas (contexto)

Uso típico::

    from roofscan.core.ingesta.carto_arba import (
        geocode_address, get_parcelas_image, get_parcel_info
    )

    lat, lon = geocode_address("Luján, Buenos Aires")
    bbox = (lon - 0.005, lat - 0.005, lon + 0.005, lat + 0.005)

    img_array, img_extent = get_parcelas_image(bbox, width=512, height=512)
    info = get_parcel_info(bbox, pixel_x=256, pixel_y=256, map_w=512, map_h=512)
"""

import io
import logging
import time
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes del servicio
# ---------------------------------------------------------------------------

WMS_BASE_URL = "https://geo.arba.gov.ar/geoserver/idera/wms"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

LAYER_PARCELA = "idera:Parcela"
LAYER_MANZANA = "idera:Manzana"
LAYERS_DEFAULT = f"{LAYER_MANZANA},{LAYER_PARCELA}"

WMS_VERSION = "1.3.0"
WMS_CRS = "EPSG:4326"

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "RoofScan/1.0 (python-requests; contacto: usuario-privado)",
})

# Caché simple en memoria (bbox_key → array)
_WMS_CACHE: dict[str, np.ndarray] = {}
_CACHE_MAX_ENTRIES = 30


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def geocode_address(address: str, country_bias: str = "ar") -> tuple[float, float]:
    """Geocodifica una dirección a coordenadas (lat, lon) usando Nominatim/OSM.

    Args:
        address: Texto de la dirección (ej. ``"Av. San Martín 1234, Luján"``).
        country_bias: Código ISO-3166 para acotar la búsqueda. Por defecto ``"ar"``
                      (Argentina).

    Returns:
        Tupla ``(lat, lon)`` en grados decimales WGS84.

    Raises:
        ValueError: Si la dirección no produce resultados.
        ConnectionError: Si Nominatim no está disponible.
    """
    if not address.strip():
        raise ValueError("La dirección no puede estar vacía.")

    params = {
        "q": address,
        "format": "json",
        "limit": 1,
        "countrycodes": country_bias,
        "addressdetails": 0,
    }

    try:
        resp = _SESSION.get(NOMINATIM_URL, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json()
    except requests.RequestException as exc:
        raise ConnectionError(
            f"No se pudo conectar al servicio de geocodificación: {exc}\n"
            "Verificá tu conexión a internet."
        ) from exc

    if not results:
        raise ValueError(
            f"No se encontraron resultados para: \"{address}\"\n"
            "Probá con una dirección más específica o incluí la localidad."
        )

    lat = float(results[0]["lat"])
    lon = float(results[0]["lon"])
    display_name = results[0].get("display_name", "")
    log.info("Geocodificado: '%s' → (%.5f, %.5f) — %s", address, lat, lon, display_name)
    return lat, lon


def get_parcelas_image(
    bbox_wgs84: tuple[float, float, float, float],
    width: int = 512,
    height: int = 512,
    layers: str = LAYERS_DEFAULT,
    use_cache: bool = True,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Descarga la imagen WMS de parcelas ARBA para el bbox dado.

    Args:
        bbox_wgs84: Bounding box en WGS84 ``(lon_min, lat_min, lon_max, lat_max)``.
        width: Ancho en píxeles de la imagen solicitada.
        height: Alto en píxeles de la imagen solicitada.
        layers: Capas WMS a solicitar (separadas por coma).
        use_cache: Si ``True``, cachea la respuesta en memoria para evitar
                   solicitudes repetidas al mismo bbox.

    Returns:
        Tupla ``(array_rgba, extent)`` donde:

        - ``array_rgba``: Array NumPy RGBA ``(H, W, 4)`` uint8 con la imagen.
        - ``extent``: Tupla ``(lon_min, lon_max, lat_min, lat_max)`` para
          usar como argumento ``extent`` en ``matplotlib.axes.imshow()``.

    Raises:
        ConnectionError: Si el servidor WMS no responde.
        ValueError: Si el servidor retorna una respuesta inesperada.
    """
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    cache_key = f"{lon_min:.5f},{lat_min:.5f},{lon_max:.5f},{lat_max:.5f},{width},{height},{layers}"

    if use_cache and cache_key in _WMS_CACHE:
        log.debug("WMS: usando caché para bbox=%s", cache_key[:40])
        return _WMS_CACHE[cache_key], (lon_min, lon_max, lat_min, lat_max)

    # En WMS 1.3.0 con CRS=EPSG:4326, el orden del bbox es: minLat,minLon,maxLat,maxLon
    wms_bbox = f"{lat_min},{lon_min},{lat_max},{lon_max}"

    params = {
        "service": "WMS",
        "version": WMS_VERSION,
        "request": "GetMap",
        "layers": layers,
        "bbox": wms_bbox,
        "width": width,
        "height": height,
        "crs": WMS_CRS,
        "format": "image/png",
        "transparent": "true",
        "styles": "",
    }

    log.info("Solicitando parcelas ARBA | bbox=%s", bbox_wgs84)
    try:
        resp = _SESSION.get(WMS_BASE_URL, params=params, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ConnectionError(
            f"No se pudo conectar al WMS de CartoARBA: {exc}\n"
            "Verificá tu conexión a internet o intentá más tarde."
        ) from exc

    content_type = resp.headers.get("Content-Type", "")
    if "image" not in content_type:
        # El servidor devolvió un error XML (ServiceException)
        error_msg = _parse_wms_error(resp.text)
        raise ValueError(
            f"El servidor WMS devolvió un error: {error_msg}\n"
            "Revisá el bbox o intentá con otra zona."
        )

    try:
        from PIL import Image
        img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        array = np.array(img, dtype=np.uint8)
    except ImportError:
        # Fallback sin Pillow: usar matplotlib
        import matplotlib.pyplot as plt
        img = plt.imread(io.BytesIO(resp.content))
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        # Asegurar canal alpha
        if img.ndim == 3 and img.shape[2] == 3:
            alpha = np.full((*img.shape[:2], 1), 255, dtype=np.uint8)
            array = np.concatenate([img, alpha], axis=2)
        else:
            array = img

    # Guardar en caché
    if use_cache:
        if len(_WMS_CACHE) >= _CACHE_MAX_ENTRIES:
            _WMS_CACHE.pop(next(iter(_WMS_CACHE)))
        _WMS_CACHE[cache_key] = array

    extent = (lon_min, lon_max, lat_min, lat_max)
    log.info("Imagen WMS recibida: %dx%d px", array.shape[1], array.shape[0])
    return array, extent


def get_parcel_info(
    bbox_wgs84: tuple[float, float, float, float],
    pixel_x: int,
    pixel_y: int,
    map_w: int,
    map_h: int,
    layer: str = LAYER_PARCELA,
) -> dict[str, Any]:
    """Obtiene información de la parcela en las coordenadas de píxel dadas.

    Usa WMS ``GetFeatureInfo`` con formato GML para obtener atributos y,
    si el servidor los provee, la geometría de la parcela.

    Args:
        bbox_wgs84: Bounding box del mapa visible ``(lon_min, lat_min, lon_max, lat_max)``.
        pixel_x: Columna del píxel clickeado (0-based).
        pixel_y: Fila del píxel clickeado (0-based).
        map_w: Ancho total del mapa en píxeles.
        map_h: Alto total del mapa en píxeles.
        layer: Capa a consultar.

    Returns:
        Dict con los atributos de la parcela. Puede incluir:

        - ``nomenclatura``: Código catastral de la parcela.
        - ``partido``, ``seccion``, ``manzana``, ``parcela``: Partes del código.
        - ``geometry_wkt``: Geometría en WKT (si el servidor la retorna).
        - ``bbox_wgs84``: Bbox aproximado de la parcela (siempre presente).
        - ``lat``, ``lon``: Coordenadas del punto clickeado.

        Retorna ``{}`` si no hay parcela en el punto dado.

    Raises:
        ConnectionError: Si el servidor no responde.
    """
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    # En WMS 1.3.0 el bbox es: minLat,minLon,maxLat,maxLon
    wms_bbox = f"{lat_min},{lon_min},{lat_max},{lon_max}"

    # Convertir píxel → coordenadas geográficas (para incluir en resultado)
    lon_click = lon_min + (pixel_x / map_w) * (lon_max - lon_min)
    lat_click = lat_max - (pixel_y / map_h) * (lat_max - lat_min)

    params = {
        "service": "WMS",
        "version": WMS_VERSION,
        "request": "GetFeatureInfo",
        "layers": layer,
        "query_layers": layer,
        "bbox": wms_bbox,
        "width": map_w,
        "height": map_h,
        "crs": WMS_CRS,
        "i": pixel_x,          # columna (WMS 1.3.0)
        "j": pixel_y,          # fila   (WMS 1.3.0)
        "info_format": "application/vnd.ogc.gml",
        "feature_count": 1,
        "styles": "",
    }

    try:
        resp = _SESSION.get(WMS_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ConnectionError(
            f"Error al consultar CartoARBA GetFeatureInfo: {exc}"
        ) from exc

    result: dict[str, Any] = {
        "lat": round(lat_click, 6),
        "lon": round(lon_click, 6),
        "bbox_wgs84": bbox_wgs84,
    }

    try:
        parsed = _parse_feature_info_gml(resp.text)
        result.update(parsed)
    except Exception as exc:
        log.warning("No se pudo parsear GetFeatureInfo GML: %s", exc)

    return result


def bbox_from_latlon(lat: float, lon: float, radius_km: float = 0.5) -> tuple:
    """Genera un bbox centrado en (lat, lon) con el radio dado en km.

    Args:
        lat: Latitud del centro en grados decimales.
        lon: Longitud del centro en grados decimales.
        radius_km: Mitad del lado del bbox en kilómetros. Por defecto 0.5 km
                   (1 km × 1 km), adecuado para ver una parcela urbana típica.

    Returns:
        Tupla ``(lon_min, lat_min, lon_max, lat_max)`` en WGS84.
    """
    # 1° de latitud ≈ 111 km; 1° de longitud ≈ 111 km × cos(lat)
    import math
    delta_lat = radius_km / 111.0
    delta_lon = radius_km / (111.0 * math.cos(math.radians(lat)))
    return (
        round(lon - delta_lon, 6),
        round(lat - delta_lat, 6),
        round(lon + delta_lon, 6),
        round(lat + delta_lat, 6),
    )


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _parse_feature_info_gml(gml_text: str) -> dict[str, Any]:
    """Extrae atributos (y opcionalmente geometría) de la respuesta GML."""
    if not gml_text.strip() or "<ServiceException" in gml_text:
        return {}

    # Intentar parsear el XML
    try:
        root = ET.fromstring(gml_text)
    except ET.ParseError:
        return {}

    result: dict[str, Any] = {}
    ns = {"gml": "http://www.opengis.net/gml"}

    # Buscar cualquier elemento Feature y sus propiedades
    # GeoServer devuelve algo como: <topp:Parcela>...<topp:nomenclatura>...
    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        tag_lower = tag.lower()

        # Atributos catastrales comunes
        if any(k in tag_lower for k in (
            "nomenclatura", "partido", "seccion", "manzana",
            "parcela", "sub", "sup", "area", "destino",
        )):
            if elem.text and elem.text.strip():
                result[tag_lower] = elem.text.strip()

        # Intentar extraer coordenadas de geometría
        if tag_lower in ("coordinates", "poslist", "pos"):
            if elem.text:
                result["_raw_coords"] = elem.text.strip()

    # Intentar construir WKT simple desde coordenadas si las tenemos
    if "_raw_coords" in result:
        try:
            wkt = _coords_to_wkt(result.pop("_raw_coords"))
            if wkt:
                result["geometry_wkt"] = wkt
        except Exception:
            result.pop("_raw_coords", None)

    return result


def _coords_to_wkt(raw: str) -> str | None:
    """Convierte una cadena de coordenadas GML a WKT aproximado."""
    try:
        parts = raw.split()
        if len(parts) < 4:
            return None
        # GML puede ser "lat lon lat lon ..." o "lon lat lon lat ..."
        coords = [(float(parts[i]), float(parts[i + 1])) for i in range(0, len(parts) - 1, 2)]
        if len(coords) < 3:
            return None
        pts = ", ".join(f"{x} {y}" for x, y in coords)
        return f"POLYGON (({pts}))"
    except (ValueError, IndexError):
        return None


def _parse_wms_error(xml_text: str) -> str:
    """Extrae el mensaje de error de una respuesta WMS ServiceException."""
    try:
        root = ET.fromstring(xml_text)
        for elem in root.iter():
            if "ServiceException" in elem.tag and elem.text:
                return elem.text.strip()
    except ET.ParseError:
        pass
    return xml_text[:200] if xml_text else "Error desconocido"
