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
2. **Cargar imagen GeoTIFF** → preview RGB automático
3. **Seleccionar motor** (Clásico o U-Net) y ajustar parámetros si se desea
4. **Detectar techos** → visualización del resultado con área total
5. **Exportar** → CSV, GeoJSON, Shapefile o PNG

### Motor U-Net

El motor U-Net requiere pesos entrenados en `data/models/unet_best.pth`. Para generarlos:
1. Ejecutar al menos 2 detecciones con el motor Clásico.
2. Guardar cada detección como feedback desde la pestaña *Validación*.
3. Hacer clic en *Reentrenar modelo U-Net*.

---

## Estructura del proyecto

```
roofscan/
├── core/
│   ├── ingesta/          # Descarga Sentinel-2, carga GeoTIFF, WMS CartoARBA
│   ├── preproceso/       # Cloud masking, reproyección, normalización
│   ├── deteccion/
│   │   ├── clasico/      # Índices espectrales + morfología
│   │   └── dl/           # U-Net, predictor, trainer
│   ├── calculo/          # Cálculo de área, geometrías
│   ├── exportacion/      # CSV, GeoTIFF, PNG
│   └── validacion/       # Métricas, feedback store
├── gui/                  # Ventana principal, mapa, resultados, validación
├── data/
│   ├── models/           # Pesos del modelo U-Net
│   ├── feedback/         # Dataset local acumulado
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
- El motor U-Net requiere al menos 2 pares de feedback para poder entrenarse.
- La descarga automática de imágenes desde la GUI es funcionalidad pendiente; por el momento se recomienda cargar un GeoTIFF ya descargado.

---

## Licencia

Uso privado. Todas las dependencias son open-source (MIT, BSD, Apache 2.0, ODbL, CC-BY).
