# PySatMeas 🛰️

**Detección y mensura de techos mediante imágenes satelitales Sentinel-2**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-Privada-red.svg)](LICENSE)

> Zona de prueba primaria: **Luján, Buenos Aires, Argentina** (~34.57°S, 59.10°O).

---

## 📋 Descripción

PySatMeas es una herramienta de escritorio para **detectar y medir superficies cubiertas (techos)** a partir de imágenes satelitales Sentinel-2. Reemplaza el flujo manual de medición en Google Maps con un pipeline automatizado, reproducible y exportable.

| Motor | Método | Cuándo usarlo |
|-------|--------|---------------|
| `DetectorClasico` | Índices espectrales (NDVI, NDBI, NDWI) + morfología OpenCV | Análisis rápido, sin GPU |
| `DetectorUNet` | Red neuronal convolucional U-Net con fine-tuning local | Máxima precisión, requiere pesos preentrenados |

### ✨ Características Principales

**Ingesta de Imágenes:**
- ✅ **Descarga automática** de imágenes Sentinel-2 L2A desde Copernicus Data Space Ecosystem (CDSE)
- ✅ **Modo offline**: carga directa de GeoTIFF ya descargado
- ✅ **Overlay catastral** con capa de parcelas CartoARBA (WMS IDERA)
- ✅ **Preproceso**: cloud masking, reproyección y normalización automáticos

**Detección y Cálculo:**
- ✅ **Dos motores intercambiables**: Clásico (índices espectrales) y U-Net (deep learning)
- ✅ **Cálculo de área en m²** con conversión píxel → metro respetando CRS (EPSG:32720)
- ✅ **Intersección parcel↔techo** para mensura por nomenclatura catastral

**Exportación y Validación:**
- ✅ **Exportación** a CSV, GeoJSON, Shapefile y PNG anotado
- ✅ **Módulo de validación**: comparación automático vs medición manual, cálculo de MAPE
- ✅ **Ciclo de feedback y reentrenamiento** incremental del modelo U-Net
- ✅ **Mensura masiva por parcela** desde CLI (`batch_mensura.py`)

---

## 🚀 Instalación

### Requisitos Previos

- Windows 10/11 (64-bit)
- **Python 3.11** o superior
- ~4 GB de espacio en disco
- Credenciales CDSE (registro gratuito en [dataspace.copernicus.eu](https://dataspace.copernicus.eu))

### Pasos de Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/federicoarielcasado/py-sat-meas.git
cd py-sat-meas

# 2. Crear y activar entorno virtual
python -m venv .venv
.venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar credenciales CDSE
copy .env.example .env
# Editar .env con tu usuario y contraseña de CDSE
```

> **Nota GDAL/rasterio en Windows:** se recomienda instalar con conda para evitar problemas con DLLs:
> ```bash
> conda install -c conda-forge gdal rasterio geopandas pyproj shapely
> ```

### Dependencias Principales

| Librería | Para qué sirve |
|----------|----------------|
| PyQt6 | Interfaz gráfica de escritorio |
| rasterio + GDAL | Lectura/escritura de raster georreferenciado |
| shapely, geopandas, pyproj | Operaciones geométricas y proyecciones CRS |
| OpenCV, scikit-image, numpy | Motor de detección clásico (morfología) |
| PyTorch + segmentation-models-pytorch | Motor U-Net (deep learning) |
| cdsetool | Descarga de imágenes desde CDSE (Copernicus) |
| matplotlib | Visualización y exportación PNG |
| pandas, SQLite | Gestión de datos y feedback store |

---

## 📖 Guía de Uso

### Caso 1: Flujo básico — Detección desde GUI

```bash
python -m roofscan
```

1. **Buscar dirección** → geocodifica y muestra el overlay de parcelas ARBA
2. **Descargar imagen Sentinel-2** → botón integrado en la GUI, búsqueda por bbox/fecha/nubosidad
3. *O* **Cargar imagen GeoTIFF** ya descargada → preview RGB automático
4. **Seleccionar motor** (Clásico o U-Net) y ajustar parámetros si se desea
5. **Detectar techos** → visualización del resultado con área total en m²
6. **Exportar** → CSV, GeoJSON, Shapefile o PNG anotado

### Caso 2: Preentrenamiento del motor U-Net con datos públicos

El motor U-Net requiere pesos en `data/models/unet_best.pth`. Para generarlos desde cero:

```bash
# 1. Generar tiles de entrenamiento (Google Open Buildings + Sentinel-2)
python scripts/prepare_tiles.py \
    --buildings data/pretrain/ARG.gpkg \
    --tiles-dir data/cache/ \
    --output-dir data/pretrain/

# 2. Entrenar el modelo
python scripts/pretrain_unet.py \
    --tiles-dir data/pretrain/ \
    --output data/models/unet_pretrained.pth
```

### Caso 3: Fine-tuning local del U-Net

1. Ejecutar al menos 2 detecciones con el motor Clásico
2. Guardar cada resultado como feedback desde la pestaña *Validación*
3. Hacer clic en **Reentrenar modelo U-Net** — el modelo se actualiza con datos locales acumulados

### Caso 4: Mensura masiva por parcela (CLI)

Para mensurar la totalidad de un partido catastral de forma automática:

```bash
# 1. Descargar catastro de parcelas una sola vez
#    Fuente: https://datos.gba.gob.ar/dataset/catastro-territorial
#    Guardar en: data/catastro/lujan.gpkg

# 2. Ejecutar mensura masiva
python scripts/batch_mensura.py \
    --image data/cache/S2A_MSIL2A_..._stacked.tif \
    --parcelas data/catastro/lujan.gpkg \
    --output data/output/mensura_lujan.csv

# Resultado: CSV con una fila por parcela (área de techo, % cubierto, nomenclatura)

# Opcional: filtrar por lista de partidas específicas
python scripts/batch_mensura.py \
    --image data/cache/S2A_MSIL2A_..._stacked.tif \
    --parcelas data/catastro/lujan.gpkg \
    --partidas mis_partidas.csv \
    --output data/output/mensura_lujan.csv
```

---

## 📐 Fundamento Teórico

### Motor Clásico — Índices Espectrales

El motor clásico combina tres índices espectrales de Sentinel-2 para separar superficies construidas del entorno:

```
NDVI  = (NIR - Red)  / (NIR + Red)     → vegetación (excluir)
NDBI  = (SWIR - NIR) / (SWIR + NIR)   → superficies construidas (incluir)
NDWI  = (Green - NIR) / (Green + NIR)  → agua (excluir)
```

Las máscaras resultantes se combinan y se refinan con operaciones morfológicas OpenCV (apertura, cierre, dilatación) para eliminar ruido y rellenar huecos.

### Motor U-Net — Segmentación Semántica

La arquitectura U-Net con encoder preentrenado (ResNet-34, ImageNet) realiza **segmentación semántica binaria** píxel a píxel:

```
Entrada: imagen multiespectral Sentinel-2 (B, G, R, NIR, SWIR) → 5 canales
Salida:  máscara binaria   0 = no techo  |  1 = techo
```

El módulo de feedback acumula ejemplos locales etiquetados y reajusta los pesos del modelo con fine-tuning supervisado.

### Cálculo de Área

```
Área [m²] = píxeles_detectados × (GSD_x × GSD_y)
```

Donde `GSD_x` y `GSD_y` son las dimensiones de píxel en metros, obtenidas de la proyección nativa de la imagen (EPSG:32720, UTM zona 20S). La intersección con geometrías catastrales se realiza con Shapely en el mismo CRS.

---

## 🧩 Arquitectura del Software

### Estructura de Directorios

```
roofscan/
├── core/
│   ├── ingesta/          # Descarga Sentinel-2, .SAFE→GeoTIFF, WMS/WFS CartoARBA
│   ├── preproceso/       # Cloud masking, reproyección, normalización
│   ├── deteccion/
│   │   ├── clasico/      # Índices espectrales + morfología OpenCV
│   │   └── dl/           # U-Net, predictor, trainer
│   ├── calculo/          # Área en m², geometrías, intersección parcel↔techo
│   ├── exportacion/      # CSV, GeoTIFF, PNG
│   └── validacion/       # Métricas (MAPE), feedback store
├── gui/                  # Ventana principal, mapa, diálogo de descarga, resultados
├── scripts/
│   ├── prepare_tiles.py  # Genera tiles .npy para preentrenamiento U-Net
│   ├── pretrain_unet.py  # Entrena U-Net con datos de edificios públicos
│   └── batch_mensura.py  # Mensura masiva de parcelas desde CLI
├── data/
│   ├── models/           # Pesos del modelo U-Net (.pth)
│   ├── feedback/         # Dataset local acumulado para fine-tuning
│   ├── pretrain/         # Tiles para preentrenamiento
│   ├── catastro/         # Archivos catastrales locales (gpkg/shp)
│   ├── output/           # Resultados de mensura masiva (CSV)
│   └── cache/            # Imágenes satelitales descargadas
└── tests/
```

### Flujo de Ejecución

```
GUI (PyQt6)
     │ dirección / GeoTIFF
     ▼
Módulo de Ingesta
     ├── CDSE API (cdsetool) → descarga .SAFE
     └── .SAFE → GeoTIFF (rasterio + GDAL)
          │
          ▼
Módulo de Preproceso
     Cloud masking + reproyección EPSG:32720 + normalización
          │
          ▼
Motor de Detección (intercambiable)
     ├── DetectorClasico: NDVI/NDBI/NDWI + morfología OpenCV
     └── DetectorUNet:    segmentación semántica (PyTorch)
          │ máscara binaria
          ▼
Módulo de Cálculo
     Área en m² + intersección con parcelas catastrales (Shapely)
          │
          ├──→ GUI: visualización + overlay catastral
          ├──→ Exportación: CSV, GeoJSON, Shapefile, PNG
          └──→ Validación: MAPE + feedback store para reentrenamiento
```

---

## 🧪 Testing

```bash
# Tests básicos (sin red ni GPU) — recomendado para CI
pytest tests/ -m "not network and not torch" -v

# Tests de red (requiere credenciales CDSE + conexión a Internet)
pytest tests/ -m network -v

# Tests de deep learning (requiere PyTorch instalado)
pytest tests/test_dl.py -v

# Todos los tests
pytest tests/ -v
```

### Marcadores de Tests

| Marcador | Descripción |
|----------|-------------|
| *(sin marcador)* | Tests unitarios locales, sin dependencias externas |
| `network` | Requiere conexión a Internet + credenciales CDSE |
| `torch` | Requiere PyTorch y pesos del modelo U-Net |

---

## 🔬 Precisión Numérica

| Aspecto | Valor / Criterio |
|---------|-----------------|
| Resolución espacial Sentinel-2 | 10 m/px (bandas RGB, NIR, SWIR) |
| Área mínima detectable (clásico) | ~30 m² (3×3 píxeles) |
| Sistema de referencia | EPSG:32720 (UTM zona 20S) |
| Métrica de validación | MAPE (Mean Absolute Percentage Error) |
| Fine-tuning U-Net | Supervisado, dataset local acumulado |

### Limitaciones Conocidas

- Techos con superficie < ~30 m² se detectan con menor precisión (límite de resolución 10 m/px).
- El motor U-Net requiere pesos preentrenados (`data/models/unet_best.pth`) para funcionar.
- La descarga de imágenes desde CDSE requiere credenciales gratuitas configuradas en `.env`.
- El catastro de parcelas debe descargarse manualmente una vez desde datos.gba.gob.ar (el WFS de ARBA no está disponible públicamente de forma estable).

---

## 📝 Changelog

### v1.0.0 (11 de Marzo de 2026)

**Implementado:**
- ✅ Módulo de ingesta: descarga Sentinel-2 L2A desde CDSE + conversión .SAFE → GeoTIFF
- ✅ Preproceso: cloud masking, reproyección y normalización
- ✅ Motor Clásico: índices espectrales (NDVI, NDBI, NDWI) + morfología OpenCV
- ✅ Motor U-Net: segmentación semántica binaria con encoder ResNet-34 preentrenado
- ✅ Pipeline de preentrenamiento con datos Google Open Buildings
- ✅ Ciclo de feedback y fine-tuning incremental
- ✅ Cálculo de área en m² con proyección nativa (EPSG:32720)
- ✅ Intersección parcel↔techo con geometrías catastrales (Shapely)
- ✅ Overlay catastral CartoARBA (WMS IDERA)
- ✅ Exportación: CSV, GeoJSON, Shapefile, PNG anotado
- ✅ Módulo de validación con cálculo de MAPE
- ✅ Mensura masiva por parcela desde CLI (`batch_mensura.py`)
- ✅ GUI PyQt6 con visor de mapa, diálogo de descarga y panel de resultados

---

## 📄 Licencia

Uso privado. Todas las dependencias son open-source (MIT, BSD, Apache 2.0, ODbL, CC-BY).

---

## 👨‍💻 Autor

**Federico Ariel Casado** — Ingeniería

- 💻 Stack técnico: Python 3.11, PyQt6, PyTorch, rasterio, GDAL, geopandas, OpenCV
- 📚 Dominio: Teledetección, segmentación semántica, análisis catastral, SIG
- 📧 federicoarielcasado@gmail.com

---

*Última actualización: 11 de Marzo de 2026*
