# RoofScan — Sprint 6: Pendientes y Guía de Continuación

**Fecha:** 2026-02-26
**Estado del proyecto:** Sprints 1–5 completos + gaps funcionales cerrados en sesión extra.

---

## ✅ Completado hasta ahora

| Sprint | Módulos clave |
|--------|---------------|
| S1 | `downloader.py`, `loader.py`, `config.py`, `.env.example`, `requirements.txt` |
| S2 | `reprojector.py`, `cloud_mask.py`, `normalizer.py`, `pipeline.py`, `raster_exporter.py` |
| S3 | `spectral_indices.py`, `morphology.py`, `area_calculator.py`, `geometry_merger.py`, GUI mínima |
| S3.5 | `carto_arba.py` (WMS IDERA), integración en GUI con geocodificación Nominatim |
| S4 | `unet.py`, `predictor.py` (tiling + Gaussian blending), `trainer.py` (RoofDataset + fine_tune) |
| S5 | `metrics.py`, `feedback_store.py`, `validation_panel.py`, `RetrainWorker` en main_window |
| Gaps | `csv_exporter.py`, selector motor Clásico/U-Net en GUI, botones CSV + Shapefile |

---

## 🔧 Sprint 6: Empaquetado Windows (PyInstaller)

### Objetivo
Distribuir RoofScan como ejecutable `.exe` sin que el usuario instale Python ni dependencias.

### Pasos detallados

#### 1. Preparar el entorno de build
```bash
# Instalar PyInstaller en el entorno de desarrollo
pip install pyinstaller

# Verificar que todas las deps están instaladas y la app corre:
python -m roofscan
```

#### 2. Instalar GDAL wheels precompilados para Windows
GDAL es la dependencia más compleja. Usar wheels de Christoph Gohlke o el canal conda:
```bash
# Opción A (recomendada): usar conda-pack
conda create -n roofscan_build python=3.11
conda activate roofscan_build
conda install -c conda-forge gdal rasterio geopandas pyproj shapely pytorch cpuonly
pip install PyQt6 segmentation-models-pytorch cdsetool python-dotenv tqdm Pillow requests

# Opción B: pip con wheels de OSGeo4W
pip install GDAL-3.x.x-cp311-cp311-win_amd64.whl   # desde https://github.com/cgohlke/geospatial-wheels
pip install rasterio-1.x.x-cp311-cp311-win_amd64.whl
```

#### 3. Crear `roofscan.spec` para PyInstaller
Archivo `roofscan.spec` en la raíz del proyecto:
```python
# roofscan.spec
import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['roofscan/__main__.py'],
    pathex=['.'],
    binaries=[
        # GDAL DLLs (ajustar paths según instalación)
        # ('C:/conda/envs/roofscan_build/Library/bin/gdal*.dll', '.'),
    ],
    datas=[
        ('roofscan/data/models/*.pth', 'roofscan/data/models'),   # pesos U-Net si existen
        ('roofscan/data/feedback/', 'roofscan/data/feedback'),
    ],
    hiddenimports=[
        'rasterio._shim',
        'rasterio.control',
        'rasterio.crs',
        'rasterio._env',
        'fiona',
        'pyproj',
        'shapely',
        'geopandas',
        'segmentation_models_pytorch',
        'timm',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=['matplotlib.tests', 'numpy.tests'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='RoofScan',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # Sin consola (app de escritorio)
    icon='docs/icon.ico',   # Crear ícono .ico si se desea
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RoofScan',
)
```

#### 4. Compilar
```bash
pyinstaller roofscan.spec --clean --noconfirm
# Resultado: dist/RoofScan/RoofScan.exe
```

#### 5. Alternativa: conda-pack (más simple)
```bash
conda activate roofscan_build
conda install conda-pack
conda pack -o roofscan_env.tar.gz

# En la PC destino:
mkdir roofscan_env
tar -xzf roofscan_env.tar.gz -C roofscan_env
roofscan_env/python -m roofscan
```

### Problemas conocidos y mitigaciones

| Problema | Mitigación |
|----------|------------|
| `rasterio` no encuentra DLLs de GDAL | Usar conda en lugar de pip; asegurar PATH incluye `Library/bin` |
| `geopandas` no encuentra `fiona` | Instalar con conda (`conda install -c conda-forge geopandas`) |
| `segmentation_models_pytorch` no encuentra `timm` | Agregar `timm` a `hiddenimports` en .spec |
| `PyQt6` platform plugin no encontrado | Copiar `platforms/qwindows.dll` junto al exe |
| `torch` muy pesado (~700 MB) | Usar build CPU-only: `--index-url https://download.pytorch.org/whl/cpu` |

---

## 📝 Documentación de usuario pendiente

Crear `docs/user_guide.md` con estas secciones:

1. **Instalación y primer uso** (requisitos: Windows 10/11 64-bit, ~4 GB espacio)
2. **Búsqueda de dirección y overlay ARBA** — cómo geocodificar y seleccionar parcelas
3. **Carga de imagen GeoTIFF** — qué formato se espera, cómo obtener de Sentinel-2
4. **Motor Clásico vs U-Net** — cuándo usar cada uno, cómo interpretar resultados
5. **Exportación** — CSV, GeoJSON, Shapefile, PNG (uso en QGIS)
6. **Ciclo de mejora continua** — validar → guardar feedback → reentrenar
7. **Limitaciones técnicas** — resolución 10 m/px, techos < 30 m² con menor precisión
8. **Credenciales CDSE** — cómo registrarse y configurar `.env`

---

## ⚠️ Deuda técnica menor

| Item | Ubicación | Prioridad |
|------|-----------|-----------|
| RF-09: edición manual de polígonos en GUI | `map_widget.py` | Baja |
| Motor U-Net integrable desde descarga directa de pesos preentrenados (Open Buildings) | `trainer.py` | Media |
| Tests de GUI (PyQt6 test con `pytest-qt`) | `tests/test_gui.py` | Baja |
| Validación de CRS automática al cargar GeoTIFF (warning si no es UTM) | `loader.py` | Baja |
| Soporte SCL (cloud mask) en el pipeline GUI | `main_window.py → AnalysisWorker` | Media |
| Descarga automática Sentinel-2 desde GUI (botón "Descargar imagen para esta parcela") | `main_window.py` | Media |

---

## 🚀 Cómo continuar en otra PC

1. Descomprimí el .zip en la carpeta de trabajo.
2. Creá el entorno virtual:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
3. Copiá `.env.example` a `.env` y completá `CDSE_USER` y `CDSE_PASSWORD`.
4. Ejecutá los tests:
   ```bash
   pytest tests/ -m "not network and not torch" -v
   ```
5. Lanzá la app:
   ```bash
   python -m roofscan
   ```

### Comandos pytest por categoría
```bash
# Tests básicos (sin red ni GPU) — siempre pasan
pytest tests/ -m "not network and not torch" -v

# Tests de red (requiere credenciales CDSE + conexión)
pytest tests/ -m network -v

# Tests de DL (requiere PyTorch instalado)
pytest tests/test_dl.py -v

# Todos los tests
pytest tests/ -v
```

---

*Documento generado automáticamente por el agente de desarrollo RoofScan — 2026-02-26*
