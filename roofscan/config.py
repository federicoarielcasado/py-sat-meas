"""Configuración global de RoofScan.

Constantes y rutas que se comparten entre todos los módulos.
No importa nada interno del proyecto.
"""

from pathlib import Path

# --- Rutas base ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = DATA_DIR / "models"
FEEDBACK_DIR = DATA_DIR / "feedback"

# --- CRS de trabajo ---
# UTM zona 20S cubre Luján correctamente; unidades métricas para cálculo de área.
CRS_WORK = "EPSG:32720"
CRS_WGS84 = "EPSG:4326"

# --- Zona de prueba por defecto: Luján, Buenos Aires, Argentina ---
# Bounding box en WGS84 (lon_min, lat_min, lon_max, lat_max)
LUJAN_BBOX_WGS84 = (-59.15, -34.60, -59.05, -34.53)

# Coordenadas del centro de Luján
LUJAN_CENTER = (-34.570, -59.105)  # (lat, lon)

# --- Parámetros de descarga Sentinel-2 ---
# Bandas a descargar: RGB + NIR + SWIR (para índices) + SCL (cloud mask)
S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]

# Colección L2A (reflectancia de superficie, cloud masking incluido)
S2_COLLECTION = "SENTINEL-2"
S2_PRODUCT_TYPE = "S2MSI2A"

# Nubosidad máxima aceptable por defecto (%)
DEFAULT_MAX_CLOUD_PCT = 20

# Resolución espacial de Sentinel-2 bandas ópticas (metros/píxel)
S2_RESOLUTION_M = 10.0

# --- Asegurar que los directorios de datos existen ---
for _dir in (CACHE_DIR, MODELS_DIR, FEEDBACK_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
