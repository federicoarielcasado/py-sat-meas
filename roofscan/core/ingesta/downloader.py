"""Módulo de descarga de imágenes Sentinel-2 desde Copernicus Data Space Ecosystem (CDSE).

Usa la STAC API para búsqueda (sin auth) y la OData API con bearer token para
descarga. La OpenSearch API usada por cdsetool 0.2.x fue dada de baja el
2026-02-02 y ya no está disponible.

Requiere credenciales CDSE gratuitas (registro en https://dataspace.copernicus.eu).
Las credenciales se leen de variables de entorno CDSE_USER y CDSE_PASSWORD,
o de un archivo .env en la raíz del proyecto.

Uso típico::

    from roofscan.core.ingesta.downloader import download_sentinel2
    from roofscan.config import LUJAN_BBOX_WGS84, CACHE_DIR

    paths = download_sentinel2(
        bbox=LUJAN_BBOX_WGS84,
        date_range=("2025-06-01", "2025-08-31"),
        output_dir=CACHE_DIR,
        max_cloud_pct=15,
    )
"""

import os
import re
import logging
import zipfile
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from roofscan.config import (
    DEFAULT_MAX_CLOUD_PCT,
    CACHE_DIR,
)

load_dotenv()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URLs públicas CDSE
# ---------------------------------------------------------------------------
_STAC_URL = (
    "https://catalogue.dataspace.copernicus.eu"
    "/stac/collections/sentinel-2-l2a/items"
)
_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
_DOWNLOAD_BASE = (
    "https://download.dataspace.copernicus.eu/odata/v1/Products"
)

# ---------------------------------------------------------------------------
# Tipos auxiliares
# ---------------------------------------------------------------------------
BBox = tuple[float, float, float, float]   # (lon_min, lat_min, lon_max, lat_max)
DateRange = tuple[str, str]                # ("YYYY-MM-DD", "YYYY-MM-DD")


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def search_sentinel2(
    bbox: BBox,
    date_range: DateRange,
    max_cloud_pct: float = DEFAULT_MAX_CLOUD_PCT,
    max_results: int = 10,
) -> list[dict]:
    """Busca escenas Sentinel-2 L2A disponibles en CDSE.

    Usa la STAC API de CDSE (no requiere autenticación).

    Args:
        bbox: Bounding box en WGS84 ``(lon_min, lat_min, lon_max, lat_max)``.
        date_range: Tupla ``("YYYY-MM-DD", "YYYY-MM-DD")`` de inicio y fin.
        max_cloud_pct: Porcentaje máximo de nubosidad aceptable (0–100).
        max_results: Número máximo de resultados a retornar.

    Returns:
        Lista de dicts con metadatos de cada escena. Cada dict contiene:
        ``{"id": str, "name": str, "date": str, "cloud_pct": float, "size_mb": float}``.

    Raises:
        ConnectionError: Si no se puede conectar a la API de CDSE.
        ValueError: Si los parámetros de entrada son inválidos.
    """
    _validate_bbox(bbox)
    _validate_date_range(date_range)

    lon_min, lat_min, lon_max, lat_max = bbox

    log.info(
        "Buscando Sentinel-2 L2A | bbox=%s | fechas=%s-%s | nube<=%s%%",
        bbox, date_range[0], date_range[1], max_cloud_pct,
    )

    try:
        resp = requests.get(
            _STAC_URL,
            params={
                "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
                "datetime": f"{date_range[0]}T00:00:00Z/{date_range[1]}T23:59:59Z",
                "limit": min(max_results * 4, 100),
                "sortby": "-datetime",
            },
            timeout=30,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ConnectionError(
            f"Error al conectar con CDSE: {exc}. "
            "Verificá tu conexión a internet."
        ) from exc

    features = resp.json().get("features", [])

    results: list[dict] = []
    for feat in features:
        props = feat.get("properties", {})
        cloud_pct = float(props.get("eo:cloud_cover", 100.0))
        if cloud_pct > max_cloud_pct:
            continue

        # Extraer ID de producto desde el asset "Product" (URL OData)
        assets = feat.get("assets", {})
        odata_href = assets.get("Product", {}).get("href", "")
        m = re.search(r"Products\(([^)]+)\)", odata_href)
        product_id = m.group(1) if m else feat.get("id", "")

        results.append({
            "id": product_id,
            "name": feat.get("id", ""),
            "date": str(props.get("datetime", ""))[:10],
            "cloud_pct": cloud_pct,
            "size_mb": 0.0,   # STAC no expone tamaño; S2 L2A ~800-1100 MB
        })

        if len(results) >= max_results:
            break

    log.info("Se encontraron %d escenas con nube<=%.0f%%.", len(results), max_cloud_pct)
    return results


def download_sentinel2(
    bbox: BBox,
    date_range: DateRange,
    output_dir: Path | None = None,
    max_cloud_pct: float = DEFAULT_MAX_CLOUD_PCT,
    max_scenes: int = 1,
) -> list[Path]:
    """Descarga escenas Sentinel-2 L2A desde CDSE vía OData.

    Descarga las escenas con menor nubosidad dentro del rango solicitado.
    El archivo se descarga como .zip y se descomprime automáticamente al
    directorio .SAFE correspondiente.

    Args:
        bbox: Bounding box en WGS84 ``(lon_min, lat_min, lon_max, lat_max)``.
        date_range: Tupla ``("YYYY-MM-DD", "YYYY-MM-DD")`` de inicio y fin.
        output_dir: Directorio de destino. Por defecto usa ``config.CACHE_DIR``.
        max_cloud_pct: Porcentaje máximo de nubosidad aceptable (0–100).
        max_scenes: Número máximo de escenas a descargar.

    Returns:
        Lista de :class:`pathlib.Path` a los directorios ``.SAFE`` descargados.

    Raises:
        EnvironmentError: Si las credenciales CDSE no están configuradas.
        ConnectionError: Si falla la comunicación con la API.
        RuntimeError: Si no se encontraron escenas o el ZIP está corrupto.
    """
    if output_dir is None:
        output_dir = CACHE_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = search_sentinel2(bbox, date_range, max_cloud_pct, max_results=max_scenes * 4)

    if not scenes:
        raise RuntimeError(
            f"No se encontraron escenas Sentinel-2 para bbox={bbox}, "
            f"fechas={date_range}, nubosidad<={max_cloud_pct}%. "
            "Probá ampliar el rango de fechas o aumentar el límite de nubosidad."
        )

    scenes_sorted = sorted(scenes, key=lambda s: s["cloud_pct"])[:max_scenes]
    token = _get_token()

    downloaded: list[Path] = []
    for scene in scenes_sorted:
        path = _download_scene(scene, output_dir, token)
        downloaded.append(path)

    return downloaded


def download_by_id(scene: dict, output_dir: Path | None = None) -> Path:
    """Descarga una escena Sentinel-2 específica dado su dict de metadatos.

    Args:
        scene: Dict con al menos ``{"id": str, "name": str}`` tal como
               lo retorna :func:`search_sentinel2`.
        output_dir: Directorio de destino. Por defecto usa ``config.CACHE_DIR``.

    Returns:
        :class:`pathlib.Path` al directorio ``.SAFE`` descargado.

    Raises:
        EnvironmentError: Si las credenciales CDSE no están configuradas.
        ConnectionError: Si falla la comunicación con la API de CDSE.
    """
    if output_dir is None:
        output_dir = CACHE_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _check_credentials()
    token = _get_token()
    return _download_scene(scene, output_dir, token)


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _get_token() -> str:
    """Obtiene bearer token desde el Identity Provider de CDSE.

    Returns:
        Token JWT como string.

    Raises:
        EnvironmentError: Si las credenciales son inválidas o faltan.
    """
    _check_credentials()
    user = os.environ["CDSE_USER"]
    password = os.environ["CDSE_PASSWORD"]

    try:
        resp = requests.post(
            _TOKEN_URL,
            data={
                "grant_type": "password",
                "username": user,
                "password": password,
                "client_id": "cdse-public",
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]
    except Exception as exc:
        raise EnvironmentError(
            f"Error al obtener token CDSE: {exc}. "
            "Verificá tus credenciales (CDSE_USER / CDSE_PASSWORD) en el archivo .env."
        ) from exc


def _download_scene(scene: dict, output_dir: Path, token: str) -> Path:
    """Descarga un producto .SAFE como ZIP y lo descomprime.

    Args:
        scene: Dict con ``{"id": str, "name": str, ...}``.
        output_dir: Directorio destino.
        token: Bearer token de autenticación CDSE.

    Returns:
        Path al directorio ``.SAFE`` descomprimido.

    Raises:
        ConnectionError: Si falla la descarga HTTP.
        RuntimeError: Si el ZIP descargado está corrupto.
    """
    scene_name = scene["name"]
    safe_path = output_dir / (scene_name + ".SAFE")

    if safe_path.exists():
        log.info("Escena ya descargada, omitiendo: %s", scene_name)
        return safe_path

    product_id = scene["id"]
    url = f"{_DOWNLOAD_BASE}({product_id})/$value"
    zip_path = output_dir / (scene_name + ".zip")

    log.info(
        "Descargando %s (nube=%.1f%%)...",
        scene_name, scene.get("cloud_pct", 0.0),
    )

    headers = {"Authorization": f"Bearer {token}"}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(zip_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        log.info(
                            "  %.1f%% (%.0f / %.0f MB)",
                            downloaded / total * 100,
                            downloaded / 1e6,
                            total / 1e6,
                        )
    except requests.RequestException as exc:
        zip_path.unlink(missing_ok=True)
        raise ConnectionError(
            f"Error al descargar {scene_name}: {exc}"
        ) from exc

    log.info("Descomprimiendo %s...", zip_path.name)
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(output_dir)
        zip_path.unlink()
    except zipfile.BadZipFile as exc:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(
            "El archivo descargado no es un ZIP válido. "
            "Las credenciales CDSE pueden haber expirado — volvé a intentarlo."
        ) from exc

    log.info("Descarga completa: %s", safe_path)
    return safe_path


def _validate_bbox(bbox: BBox) -> None:
    """Valida que el bounding box tenga formato y valores correctos."""
    if len(bbox) != 4:
        raise ValueError(
            "bbox debe tener exactamente 4 valores: (lon_min, lat_min, lon_max, lat_max)"
        )
    lon_min, lat_min, lon_max, lat_max = bbox
    if lon_min >= lon_max:
        raise ValueError(f"lon_min ({lon_min}) debe ser menor que lon_max ({lon_max})")
    if lat_min >= lat_max:
        raise ValueError(f"lat_min ({lat_min}) debe ser menor que lat_max ({lat_max})")
    if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
        raise ValueError("Las longitudes deben estar entre -180 y 180")
    if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
        raise ValueError("Las latitudes deben estar entre -90 y 90")


def _validate_date_range(date_range: DateRange) -> None:
    """Valida que el rango de fechas sea coherente."""
    if len(date_range) != 2:
        raise ValueError("date_range debe ser una tupla de 2 fechas: (inicio, fin)")
    try:
        start = datetime.strptime(date_range[0], "%Y-%m-%d")
        end = datetime.strptime(date_range[1], "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"Las fechas deben estar en formato YYYY-MM-DD. Error: {exc}"
        ) from exc
    if start > end:
        raise ValueError(
            f"La fecha de inicio ({date_range[0]}) debe ser anterior "
            f"a la de fin ({date_range[1]})"
        )


def _check_credentials() -> None:
    """Verifica que las credenciales CDSE estén disponibles."""
    user = os.environ.get("CDSE_USER", "").strip()
    password = os.environ.get("CDSE_PASSWORD", "").strip()
    if not user or not password:
        raise EnvironmentError(
            "Credenciales CDSE no configuradas. "
            "Copiá .env.example como .env y completá CDSE_USER y CDSE_PASSWORD. "
            "Registro gratuito en: https://dataspace.copernicus.eu"
        )
