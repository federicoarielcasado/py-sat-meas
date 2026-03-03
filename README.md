# Py-sat-meas — Detección de Superficies Cubiertas con Imágenes Satelitales

Herramienta de escritorio para detectar y medir superficies cubiertas (techos) a partir de imágenes satelitales Sentinel-2. Reemplaza el flujo manual de medición en Google Maps con un pipeline automatizado, reproducible y exportable.

**Zona de prueba primaria:** Luján, Buenos Aires, Argentina (~34.57°S, 59.10°O).

---

## ¿Qué hace?

1. El usuario ingresa una dirección o carga una imagen GeoTIFF.
2. La app procesa y segmenta la imagen satelital automáticamente.
3. Exporta las áreas detectadas en m², junto con geometrías GeoJSON/Shapefile y visualización anotada.

---

## Características principales

- **Descarga automática** de imágenes Sentinel-2 L2A desde Copernicus Data Space Ecosystem (CDSE)
- **Modo offline**: carga directa de GeoTIFF ya descargado
- **Overlay catastral** con capa de parcelas CartoARBA (WMS IDERA)
- **Dos motores de detección intercambiables:**
  - *Clásico*: índices espectrales (NDVI, NDBI, NDWI) + morfología OpenCV
  - *U-Net*: red neuronal convolucional con capacidad de fine-tuning local
- **Cálculo de área** en m² con conversión píxel → metro respetando CRS (EPSG:32720)
- **Exportación** a CSV, GeoJSON, Shapefile y PNG anotado
- **Módulo de validación**: comparación automático vs medición manual, cálculo de MAPE
- **Ciclo de feedback y reentrenamiento** incremental del modelo U-Net

---

## Stack tecnológico

| Componente | Librería |
|------------|----------|
| GUI | PyQt6 |
| Raster I/O | rasterio + GDAL |
| Geometrías | shapely, geopandas, pyproj |
| CV clásico | OpenCV, scikit-image, numpy |
| Deep Learning | PyTorch + segmentation-models-pytorch |
| Descarga satelital | cdsetool |
| Visualización | matplotlib |
| Datos | pandas, SQLite |

---

## Instalación

### Requisitos
- Windows 10/11 (64-bit)
- Python 3.11+
- ~4 GB de espacio en disco

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/federicoarielcasado/py-sat-meas.git
cd py-sat-meas

# 2. Crear y activar entorno virtual
python -m venv .venv
.venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar credenciales CDSE (registro gratuito en https://dataspace.copernicus.eu)
copy .env.example .env
# Editar .env con tu usuario y contraseña de CDSE
```

> **Nota sobre GDAL/rasterio en Windows:** se recomienda instalar con conda para evitar problemas con DLLs:
> ```bash
> conda install -c conda-forge gdal rasterio geopandas pyproj shapely
> ```

---

## Uso

```bash
python -m roofscan
```

### Flujo básico

1. **Buscar dirección** → geocodifica y muestra el overlay de parcelas ARBA
2. **Descargar imagen Sentinel-2** → botón integrado en la GUI, búsqueda por bbox/fecha/nubosidad, descarga y conversión automática
3. **O cargar imagen GeoTIFF** ya descargada → preview RGB automático
4. **Seleccionar motor** (Clásico o U-Net) y ajustar parámetros si se desea
5. **Detectar techos** → visualización del resultado con área total
6. **Exportar** → CSV, GeoJSON, Shapefile o PNG

### Motor U-Net

El motor U-Net requiere pesos entrenados en `data/models/unet_best.pth`. Hay dos formas de generarlos:

**Opción A — Preentrenamiento con datos públicos (recomendado como punto de partida):**
```bash
# 1. Generar tiles de entrenamiento desde Google Open Buildings + imágenes Sentinel-2
python scripts/prepare_tiles.py \
    --buildings data/pretrain/ARG.gpkg \
    --tiles-dir data/cache/ \
    --output-dir data/pretrain/

# 2. Entrenar el modelo
python scripts/pretrain_unet.py \
    --tiles-dir data/pretrain/ \
    --output data/models/unet_pretrained.pth
```

**Opción B — Fine-tuning con datos locales:**
1. Ejecutar al menos 2 detecciones con el motor Clásico.
2. Guardar cada detección como feedback desde la pestaña *Validación*.
3. Hacer clic en *Reentrenar modelo U-Net*.

### Mensura masiva por parcela

Para mensurar la totalidad de un partido catastral (ej: Luján) de forma automática:

```bash
# 1. Descargar catastro de parcelas (una sola vez)
#    Fuente: https://datos.gba.gob.ar/dataset/catastro-territorial
#    Guardar en: data/catastro/lujan.gpkg

# 2. Ejecutar mensura masiva
python scripts/batch_mensura.py \
    --image data/cache/S2A_MSIL2A_..._stacked.tif \
    --parcelas data/catastro/lujan.gpkg \
    --output data/output/mensura_lujan.csv

# Resultado: CSV con una fila por parcela (área de techo, % cubierto, nomenclatura)
```

También admite una lista de partidas específicas con `--partidas mis_partidas.csv`.

---

## Estructura del proyecto

```
roofscan/
├── core/
│   ├── ingesta/          # Descarga Sentinel-2, conversión .SAFE→GeoTIFF, WMS/WFS CartoARBA
│   ├── preproceso/       # Cloud masking, reproyección, normalización
│   ├── deteccion/
│   │   ├── clasico/      # Índices espectrales + morfología
│   │   └── dl/           # U-Net, predictor, trainer
│   ├── calculo/          # Cálculo de área, geometrías, intersección parcel↔techo
│   ├── exportacion/      # CSV, GeoTIFF, PNG
│   └── validacion/       # Métricas, feedback store
├── gui/                  # Ventana principal, mapa, diálogo de descarga, resultados
├── scripts/
│   ├── prepare_tiles.py  # Genera tiles .npy para preentrenamiento U-Net
│   ├── pretrain_unet.py  # Entrena U-Net con tiles de edificios públicos
│   └── batch_mensura.py  # Mensura masiva de parcelas desde CLI
├── data/
│   ├── models/           # Pesos del modelo U-Net
│   ├── feedback/         # Dataset local acumulado
│   ├── pretrain/         # Tiles para preentrenamiento
│   ├── catastro/         # Archivos catastrales locales (gpkg/shp)
│   ├── output/           # Resultados de mensura masiva
│   └── cache/            # Imágenes satelitales descargadas
└── tests/
```

---

## Tests

```bash
# Tests básicos (sin red ni GPU)
pytest tests/ -m "not network and not torch" -v

# Tests de red (requiere credenciales CDSE + conexión)
pytest tests/ -m network -v

# Tests de deep learning (requiere PyTorch)
pytest tests/test_dl.py -v
```

---

## Distribución para Windows

Se incluyen dos métodos para empaquetar la aplicación y distribuirla sin necesidad de instalar Python:

### Opción A — PyInstaller (ejecutable nativo)

```bash
# Activar entorno con todas las dependencias instaladas
pip install pyinstaller

# Compilar
build_windows.bat
# Resultado: dist\RoofScan\RoofScan.exe
```

Comprime la carpeta `dist\RoofScan\` y compártela. El usuario solo necesita extraerla y ejecutar `RoofScan.exe`.

### Opción B — conda-pack (entorno completo)

Más robusto con dependencias nativas (GDAL, rasterio). Produce un ZIP más grande (~2–4 GB).

```bash
conda activate roofscan
build_conda.bat
# Resultado: roofscan_env.zip + launch_roofscan.bat
```

---

## Limitaciones conocidas

- Sentinel-2 tiene resolución de 10 m/px. Techos con superficie < ~30 m² se detectan con menor precisión.
- El motor U-Net requiere pesos preentrenados (`data/models/unet_best.pth`) para funcionar. Ver sección *Motor U-Net*.
- La descarga de imágenes desde CDSE requiere credenciales gratuitas configuradas en `.env`.
- Para la mensura masiva, el catastro de parcelas debe descargarse manualmente una vez desde datos.gba.gob.ar (el WFS de ARBA no está disponible públicamente de forma estable).

---

## Licencia

Uso privado. Todas las dependencias son open-source (MIT, BSD, Apache 2.0, ODbL, CC-BY).
