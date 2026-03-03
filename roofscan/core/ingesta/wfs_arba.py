"""Obtención de polígonos catastrales de parcelas ARBA/IDERA.

Implementa tres estrategias en orden de preferencia:

Estrategia 1 — Archivo local (más confiable, funciona offline):
    El usuario descarga el catastro una vez y lo provee al script.
    Fuentes recomendadas:
      · Datos Abiertos Buenos Aires:
        https://datos.gba.gob.ar/dataset/catastro-territorial
      · IGN — Capas SIG Argentina:
        https://www.ign.gob.ar/NuestrasActividades/InformacionGeoespacial/CapasIG
    Formatos soportados: GeoPackage (.gpkg), Shapefile (.shp), GeoJSON.

Estrategia 2 — WFS automático (requiere conexión):
    Intenta varios endpoints WFS conocidos de ARBA, IDERA y provincia de BA.
    El servidor de ARBA (geo.arba.gov.ar) tiene WFS deshabilitado al momento
    de escritura de este módulo; los endpoints alternativos pueden funcionar.

Estrategia 3 — Escaneo WMS GetFeatureInfo en grilla (fallback lento):
    Lanza consultas GetFeatureInfo sobre una grilla de puntos dentro del bbox,
    deduplica por nomenclatura y reconstruye polígonos aproximados desde GML.
    Útil para zonas pequeñas o listas de partidas puntuales.

────────────────────────────────────────────────────
 Uso típico
────────────────────────────────────────────────────

  from roofscan.core.ingesta.wfs_arba import get_parcelas

  # Con archivo local (recomendado para el partido completo):
  gdf = get_parcelas(
      bbox_wgs84=(-59.15, -34.70, -58.90, -34.45),
      local_file="data/catastro/lujan.gpkg",
  )

  # Con WFS automático:
  gdf = get_parcelas(
      bbox_wgs84=(-59.15, -34.70, -58.90, -34.45),
  )

  # Con filtro de partidas (nomenclaturas):
  gdf = get_parcelas(
      bbox_wgs84=(-59.15, -34.70, -58.90, -34.45),
      local_file="data/catastro/lujan.gpkg",
      nomenclaturas=["067-A-1-23", "067-A-1-24"],
  )
"""

import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Endpoints WFS conocidos (se prueban en orden)
# ---------------------------------------------------------------------------

_WFS_ENDPOINTS = [
    # OWS endpoint de ARBA (a veces responde cuando /idera/wfs no lo hace)
    "https://geo.arba.gov.ar/geoserver/ows",
    # Endpoint directo idera (conocido como deshabilitado pero se intenta)
    "https://geo.arba.gov.ar/geoserver/idera/wfs",
    # IDERA nacional
    "https://www.idera.gob.ar/geoserver/ows",
    # Infraestructura de datos espaciales Buenos Aires
    "https://ide.gba.gob.ar/geoserver/ows",
]

_WFS_TYPENAME = "idera:Parcela"
_WFS_OUTPUT_FORMAT = "application/json"
_WFS_TIMEOUT = 30          # segundos por request
_WFS_MAX_FEATURES = 50000  # límite de features por pedido

_WMS_BASE_URL = "https://geo.arba.gov.ar/geoserver/idera/wms"
_WMS_LAYER = "idera:Parcela"

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "py-sat-meas/1.0 (python-requests; uso académico)",
})


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def get_parcelas(
    bbox_wgs84: tuple[float, float, float, float] | None = None,
    local_file: str | Path | None = None,
    wfs_url: str | None = None,
    nomenclaturas: list[str] | None = None,
    scan_step_deg: float = 0.003,
) -> "GeoDataFrame":
    """Obtiene parcelas catastrales usando la mejor estrategia disponible.

    Prioridad: archivo local → WFS → escaneo WMS.

    Args:
        bbox_wgs84: Bounding box ``(lon_min, lat_min, lon_max, lat_max)`` en WGS84.
                    Requerido si no se provee ``local_file`` o si se quiere
                    filtrar el archivo local por área.
        local_file: Ruta a un archivo catastral local (.gpkg, .shp, .geojson).
                    Si se provee, es la opción más rápida y confiable.
        wfs_url:    URL WFS personalizada. Si se provee, se usa directamente
                    en lugar de probar los endpoints predefinidos.
        nomenclaturas: Lista de nomenclaturas catastrales a filtrar
                       (ej. ``["067-A-1-23", "067-A-1-24"]``).
                       Si ``None`` se retornan todas las parcelas del bbox.
        scan_step_deg: Paso de la grilla en grados para el escaneo WMS
                       (solo se usa como fallback). Default 0.003° ≈ 330 m.

    Returns:
        :class:`geopandas.GeoDataFrame` con CRS WGS84 (EPSG:4326) y las
        columnas disponibles según la fuente (al menos ``geometry`` y,
        cuando exista, ``nomenclatura``).

    Raises:
        ValueError: Si no se puede obtener parcelas por ningún medio.
        ImportError: Si geopandas no está instalado.
    """
    _check_geopandas()

    # --- Estrategia 1: archivo local ---
    if local_file is not None:
        log.info("Estrategia 1: cargando catastro desde archivo local: %s", local_file)
        gdf = _from_local_file(local_file, bbox_wgs84, nomenclaturas)
        if len(gdf) > 0:
            log.info("  %d parcelas cargadas desde archivo local.", len(gdf))
            return gdf
        log.warning("  El archivo local no retornó parcelas para el bbox/filtro dado.")

    # --- Estrategia 2: WFS ---
    if bbox_wgs84 is not None:
        endpoints = [wfs_url] if wfs_url else _WFS_ENDPOINTS
        for url in endpoints:
            log.info("Estrategia 2: intentando WFS en %s", url)
            try:
                gdf = _from_wfs(url, bbox_wgs84, nomenclaturas)
                if len(gdf) > 0:
                    log.info("  %d parcelas obtenidas via WFS.", len(gdf))
                    return gdf
            except Exception as exc:
                log.warning("  WFS %s falló: %s", url, exc)

        # --- Estrategia 3: escaneo WMS (fallback) ---
        log.info(
            "Estrategia 3: escaneo WMS GetFeatureInfo (paso=%.4f°). Puede tardar varios minutos…",
            scan_step_deg,
        )
        try:
            gdf = _scan_wms(bbox_wgs84, step_deg=scan_step_deg, nomenclaturas=nomenclaturas)
            if len(gdf) > 0:
                log.info("  %d parcelas obtenidas via escaneo WMS.", len(gdf))
                return gdf
        except Exception as exc:
            log.warning("  Escaneo WMS falló: %s", exc)

    raise ValueError(
        "No se pudieron obtener parcelas por ningún medio disponible.\n"
        "\n"
        "Opciones para resolver esto:\n"
        "  1. Descargá el catastro del partido de Luján desde:\n"
        "       https://datos.gba.gob.ar/dataset/catastro-territorial\n"
        "     y usá --parcelas /ruta/lujan.gpkg en el script.\n"
        "\n"
        "  2. Si tenés acceso a un WFS personalizado, usá --wfs-url <URL>.\n"
        "\n"
        "  3. Ampliá el bbox o reducí el paso de grilla con --scan-step."
    )


def get_parcelas_by_nomenclaturas(
    nomenclaturas: list[str],
    bbox_wgs84: tuple[float, float, float, float],
    local_file: str | Path | None = None,
) -> "GeoDataFrame":
    """Obtiene un subconjunto de parcelas filtrando por lista de nomenclaturas.

    Conveniente para el flujo "tengo una lista de partidas → quiero sus polígonos".

    Args:
        nomenclaturas: Lista de códigos de nomenclatura catastral.
        bbox_wgs84: Bbox de búsqueda (debe abarcar todas las parcelas de la lista).
        local_file: Archivo catastral local para búsqueda (recomendado).

    Returns:
        GeoDataFrame con las parcelas encontradas (puede ser un subconjunto
        si alguna nomenclatura no se encontró).
    """
    return get_parcelas(
        bbox_wgs84=bbox_wgs84,
        local_file=local_file,
        nomenclaturas=nomenclaturas,
    )


# ---------------------------------------------------------------------------
# Estrategia 1: archivo local
# ---------------------------------------------------------------------------

def _from_local_file(
    filepath: str | Path,
    bbox_wgs84: tuple | None,
    nomenclaturas: list[str] | None,
) -> "GeoDataFrame":
    import geopandas as gpd

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Archivo catastral no encontrado: {filepath}\n"
            "Descargalo desde https://datos.gba.gob.ar/dataset/catastro-territorial"
        )

    log.info("  Leyendo %s…", filepath.name)

    # Para archivos grandes, usar bbox como filtro de lectura si geopandas lo soporta
    if bbox_wgs84 is not None:
        lon_min, lat_min, lon_max, lat_max = bbox_wgs84
        try:
            gdf = gpd.read_file(filepath, bbox=(lon_min, lat_min, lon_max, lat_max))
        except Exception:
            # Fallback: leer todo y filtrar a mano
            gdf = gpd.read_file(filepath)
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs("EPSG:4326")
            gdf = gdf.cx[lon_min:lon_max, lat_min:lat_max]
    else:
        gdf = gpd.read_file(filepath)

    if gdf.crs is None:
        log.warning("  Archivo sin CRS definido. Asignando WGS84.")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # Filtrar por nomenclaturas si se piden
    if nomenclaturas:
        gdf = _filter_by_nomenclaturas(gdf, nomenclaturas)

    return gdf.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Estrategia 2: WFS
# ---------------------------------------------------------------------------

def _from_wfs(
    wfs_url: str,
    bbox_wgs84: tuple,
    nomenclaturas: list[str] | None,
) -> "GeoDataFrame":
    import geopandas as gpd
    import io

    lon_min, lat_min, lon_max, lat_max = bbox_wgs84

    # Construir CQL filter
    cql_parts = []
    if nomenclaturas:
        nom_list = ",".join(f"'{n}'" for n in nomenclaturas)
        cql_parts.append(f"nomenclatura IN ({nom_list})")

    params: dict[str, Any] = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": _WFS_TYPENAME,
        "outputFormat": _WFS_OUTPUT_FORMAT,
        "count": _WFS_MAX_FEATURES,
        "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max},EPSG:4326",
    }
    if cql_parts:
        params["CQL_FILTER"] = " AND ".join(cql_parts)

    resp = _SESSION.get(wfs_url, params=params, timeout=_WFS_TIMEOUT)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "xml" in content_type.lower() and "exception" in resp.text.lower():
        raise ConnectionError(f"WFS retornó ServiceException: {resp.text[:300]}")

    gdf = gpd.read_file(io.BytesIO(resp.content))
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    return gdf.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Estrategia 3: escaneo WMS GetFeatureInfo
# ---------------------------------------------------------------------------

def _scan_wms(
    bbox_wgs84: tuple,
    step_deg: float = 0.003,
    nomenclaturas: list[str] | None = None,
) -> "GeoDataFrame":
    """Escanea el bbox con GetFeatureInfo en grilla y deduplica por nomenclatura.

    Limitado a áreas pequeñas por ser lento. Para un partido completo como Luján
    usar step_deg >= 0.003 (≈330 m) es razonable, aunque perderá parcelas
    muy pequeñas o irregularmente distribuidas.
    """
    import geopandas as gpd
    from shapely.geometry import box, Point
    from shapely import wkt as shapely_wkt

    lon_min, lat_min, lon_max, lat_max = bbox_wgs84

    import numpy as np
    lons = np.arange(lon_min + step_deg / 2, lon_max, step_deg)
    lats = np.arange(lat_min + step_deg / 2, lat_max, step_deg)
    total_pts = len(lons) * len(lats)
    log.info("  Grilla WMS: %d × %d = %d puntos", len(lons), len(lats), total_pts)

    if total_pts > 10_000:
        log.warning(
            "  La grilla tiene %d puntos. Considerá aumentar --scan-step "
            "o usar un archivo local para el catastro.", total_pts
        )

    # Mapa temporal bbox para GetFeatureInfo (ventana pequeña alrededor de cada punto)
    half = step_deg / 2
    map_w = map_h = 64   # tamaño del mapa WMS virtual (px)
    px_center = map_w // 2

    seen: dict[str, dict] = {}  # nomenclatura → feature dict
    n_requests = 0

    for lat in lats:
        for lon in lons:
            pt_bbox = (lon - half, lat - half, lon + half, lat + half)
            try:
                info = _wms_get_feature_info(pt_bbox, px_center, px_center, map_w, map_h)
            except Exception as exc:
                log.debug("  GetFeatureInfo falló en (%.4f, %.4f): %s", lat, lon, exc)
                continue

            if not info:
                continue

            nom = info.get("nomenclatura") or info.get("parcela") or f"_{lat:.4f}_{lon:.4f}"
            if nom not in seen:
                seen[nom] = info
                if nomenclaturas and nom not in nomenclaturas:
                    continue

            n_requests += 1
            if n_requests % 100 == 0:
                log.info("  Progreso escaneo: %d/%d puntos | %d parcelas únicas", n_requests, total_pts, len(seen))

            time.sleep(0.05)  # throttle para no sobrecargar el servidor

    log.info("  Escaneo WMS completo: %d parcelas únicas encontradas.", len(seen))

    if not seen:
        return gpd.GeoDataFrame(columns=["nomenclatura", "geometry"], crs="EPSG:4326")

    # Construir GeoDataFrame desde los resultados
    rows = []
    for nom, info in seen.items():
        geom = None
        wkt_str = info.get("geometry_wkt")
        if wkt_str:
            try:
                geom = shapely_wkt.loads(wkt_str)
            except Exception:
                pass
        if geom is None:
            # Punto aproximado si no hay geometría
            geom = Point(info.get("lon", 0), info.get("lat", 0))

        row = {k: v for k, v in info.items() if k not in ("geometry_wkt", "lat", "lon", "bbox_wgs84", "_raw_coords")}
        row["geometry"] = geom
        rows.append(row)

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    if nomenclaturas:
        gdf = _filter_by_nomenclaturas(gdf, nomenclaturas)

    return gdf.reset_index(drop=True)


def _wms_get_feature_info(
    bbox: tuple, px: int, py: int, map_w: int, map_h: int
) -> dict[str, Any]:
    """Lanza un GetFeatureInfo WMS y retorna los atributos de la parcela."""
    lon_min, lat_min, lon_max, lat_max = bbox
    wms_bbox = f"{lat_min},{lon_min},{lat_max},{lon_max}"

    params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetFeatureInfo",
        "layers": _WMS_LAYER,
        "query_layers": _WMS_LAYER,
        "bbox": wms_bbox,
        "width": map_w,
        "height": map_h,
        "crs": "EPSG:4326",
        "i": px,
        "j": py,
        "info_format": "application/vnd.ogc.gml",
        "feature_count": 1,
        "styles": "",
    }

    resp = _SESSION.get(_WMS_BASE_URL, params=params, timeout=10)
    resp.raise_for_status()
    return _parse_gml(resp.text)


def _parse_gml(gml_text: str) -> dict[str, Any]:
    """Extrae atributos de una respuesta GML de GetFeatureInfo."""
    if not gml_text.strip() or "<ServiceException" in gml_text:
        return {}
    try:
        root = ET.fromstring(gml_text)
    except ET.ParseError:
        return {}

    result: dict[str, Any] = {}
    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        tag_lower = tag.lower()
        if any(k in tag_lower for k in (
            "nomenclatura", "partido", "seccion", "manzana",
            "parcela", "sub", "sup", "area", "destino",
        )):
            if elem.text and elem.text.strip():
                result[tag_lower] = elem.text.strip()
        if tag_lower in ("coordinates", "poslist", "pos") and elem.text:
            result["_raw_coords"] = elem.text.strip()

    if "_raw_coords" in result:
        try:
            raw = result.pop("_raw_coords")
            parts = raw.split()
            if len(parts) >= 4:
                coords = [(float(parts[i]), float(parts[i + 1]))
                          for i in range(0, len(parts) - 1, 2)]
                if len(coords) >= 3:
                    pts = ", ".join(f"{x} {y}" for x, y in coords)
                    result["geometry_wkt"] = f"POLYGON (({pts}))"
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_by_nomenclaturas(gdf: "GeoDataFrame", nomenclaturas: list[str]) -> "GeoDataFrame":
    """Filtra un GeoDataFrame por lista de nomenclaturas."""
    nom_cols = [c for c in gdf.columns if "nomenclatura" in c.lower() or c.lower() == "parcela"]
    if not nom_cols:
        log.warning("No se encontró columna de nomenclatura en el GeoDataFrame. Sin filtrar.")
        return gdf
    col = nom_cols[0]
    return gdf[gdf[col].isin(nomenclaturas)]


def _check_geopandas() -> None:
    try:
        import geopandas  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "geopandas no está instalado. Ejecutá: pip install geopandas"
        ) from exc
