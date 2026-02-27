"""Módulo de descarga de imágenes Sentinel-2 desde Copernicus Data Space Ecosystem (CDSE).

Requiere credenciales CDSE gratuitas (registro en https://dataspace.copernicus.eu).
Las credenciales se leen de variables de entorno CDSE_USER y CDSE_PASSWORD,
o de un archivo .env en la raíz del proyecto.

Uso típico::

    from roofscan.core.ingesta.downloader import download_sentinel2
    from roofscan.config import LUJAN_BBOX_WGS84, CACHE_DIR

    paths = download_sentinel2(
        bbox=LUJAN_BBOX_WGS84,
        date_range=("2024-01-01", "2024-03-31"),
        output_dir=CACHE_DIR,
        max_cloud_pct=15,
    )
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from roofscan.config import (
    S2_PRODUCT_TYPE,
    DEFAULT_MAX_CLOUD_PCT,
    CACHE_DIR,
)

load_dotenv()
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tipos auxiliares
# ---------------------------------------------------------------------------
BBox = tuple[float, float, float, float]  # (lon_min, lat_min, lon_max, lat_max)
DateRange = tuple[str, str]               # ("YYYY-MM-DD", "YYYY-MM-DD")


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def search_sentinel2(
    bbox: BBox,
    date_range: DateRange,
    max_cloud_pct: float = DEFAULT_MAX_CLOUD_PCT,
    max_results: int = 10,
) -> list[dict]:
    """Busca escenas Sentinel-2 L2A disponibles en CDSE para la zona y fechas dadas.

    Args:
        bbox: Bounding box en WGS84 ``(lon_min, lat_min, lon_max, lat_max)``.
        date_range: Tupla ``("YYYY-MM-DD", "YYYY-MM-DD")`` de inicio y fin.
        max_cloud_pct: Porcentaje máximo de nubosidad aceptable (0–100).
        max_results: Número máximo de resultados a retornar.

    Returns:
        Lista de dicts con metadatos de cada escena. Cada dict contiene al menos
        ``{"id": str, "name": str, "date": str, "cloud_pct": float, "size_mb": float}``.

    Raises:
        EnvironmentError: Si las credenciales CDSE no están configuradas.
        ConnectionError: Si no se puede conectar a la API de CDSE.
        ValueError: Si los parámetros de entrada son inválidos.
    """
    _validate_bbox(bbox)
    _validate_date_range(date_range)
    _check_credentials()

    try:
        from cdsetool.query import query_features
        from cdsetool.credentials import Credentials
    except ImportError as exc:
        raise ImportError(
            "La librería 'cdsetool' no está instalada. "
            "Ejecutá: pip install cdsetool"
        ) from exc

    user = os.environ["CDSE_USER"]
    password = os.environ["CDSE_PASSWORD"]

    log.info(
        "Buscando Sentinel-2 L2A | bbox=%s | fechas=%s–%s | nube<=%s%%",
        bbox, date_range[0], date_range[1], max_cloud_pct,
    )

    try:
        credentials = Credentials(user, password)
        lon_min, lat_min, lon_max, lat_max = bbox
        aoi_wkt = (
            f"POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},"
            f"{lon_max} {lat_max},{lon_min} {lat_max},{lon_min} {lat_min}))"
        )

        features = list(
            query_features(
                "Sentinel2",
                {
                    "startDate": date_range[0],
                    "completionDate": date_range[1],
                    "processingLevel": "S2MSI2A",
                    "cloudCover": f"[0,{int(max_cloud_pct)}]",
                    "geometry": aoi_wkt,
                },
            )
        )[:max_results]

    except Exception as exc:
        raise ConnectionError(
            f"Error al conectar con CDSE: {exc}. "
            "Verificá tu conexión a internet y tus credenciales."
        ) from exc

    results = []
    for feat in features:
        props = feat.get("properties", {})
        results.append({
            "id": feat.get("id", ""),
            "name": props.get("title", ""),
            "date": props.get("startDate", "")[:10],
            "cloud_pct": props.get("cloudCover", 0.0),
            "size_mb": props.get("services", {}).get("download", {}).get("size", 0) / 1e6,
        })

    log.info("Se encontraron %d escenas.", len(results))
    return results


def download_sentinel2(
    bbox: BBox,
    date_range: DateRange,
    output_dir: Path | None = None,
    max_cloud_pct: float = DEFAULT_MAX_CLOUD_PCT,
    max_scenes: int = 1,
) -> list[Path]:
    """Descarga escenas Sentinel-2 L2A desde CDSE.

    Descarga la(s) escena(s) con menor nubosidad dentro del rango solicitado.

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
        RuntimeError: Si no se encontraron escenas para los parámetros dados.
        ValueError: Si los parámetros de entrada son inválidos.
    """
    if output_dir is None:
        output_dir = CACHE_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = search_sentinel2(bbox, date_range, max_cloud_pct, max_results=max_scenes * 3)

    if not scenes:
        raise RuntimeError(
            f"No se encontraron escenas Sentinel-2 para bbox={bbox}, "
            f"fechas={date_range}, nubosidad<={max_cloud_pct}%. "
            "Probá ampliar el rango de fechas o aumentar el límite de nubosidad."
        )

    # Ordenar por nubosidad ascendente y tomar las primeras max_scenes
    scenes_sorted = sorted(scenes, key=lambda s: s["cloud_pct"])[:max_scenes]

    try:
        from cdsetool.download import download_feature
        from cdsetool.credentials import Credentials
    except ImportError as exc:
        raise ImportError("La librería 'cdsetool' no está instalada.") from exc

    user = os.environ["CDSE_USER"]
    password = os.environ["CDSE_PASSWORD"]
    credentials = Credentials(user, password)

    downloaded_paths: list[Path] = []
    for scene in scenes_sorted:
        scene_id = scene["id"]
        scene_name = scene["name"]
        dest = output_dir / scene_name

        if dest.exists():
            log.info("Escena ya descargada, omitiendo: %s", scene_name)
            downloaded_paths.append(dest)
            continue

        log.info(
            "Descargando %s (nube=%.1f%%, ~%.0f MB)...",
            scene_name, scene["cloud_pct"], scene["size_mb"],
        )
        try:
            download_feature(scene_id, str(output_dir), credentials)
            downloaded_paths.append(dest)
            log.info("Descarga completa: %s", dest)
        except Exception as exc:
            raise ConnectionError(
                f"Error al descargar la escena {scene_name}: {exc}"
            ) from exc

    return downloaded_paths


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _validate_bbox(bbox: BBox) -> None:
    """Valida que el bounding box tenga formato y valores correctos."""
    if len(bbox) != 4:
        raise ValueError("bbox debe tener exactamente 4 valores: (lon_min, lat_min, lon_max, lat_max)")
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
            f"La fecha de inicio ({date_range[0]}) debe ser anterior a la de fin ({date_range[1]})"
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
