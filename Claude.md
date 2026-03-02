# Claude.md — Detector de Superficies Cubiertas con Imágenes Satelitales
**Versión:** 1.0 | **Fecha:** 2026-02-26 | **Autor del contexto:** Fede

---

## 0) Meta y Contexto General

**Nombre del proyecto:** `RoofScan` — Detección automática de superficies cubiertas  
**Dominio:** Computer Vision · GIS · Ingeniería Civil aplicada  
**Contexto operativo:** Aplicación de escritorio Windows, distribuida privadamente entre compañeros de trabajo. Reemplaza el flujo manual actual: carto ARBA (parcelas) + Google Maps (medición visual) con una herramienta automatizada, reproducible y cuantificable.

**Zona de prueba primaria:** Localidad de Luján, Buenos Aires, Argentina (~34.57°S, 59.10°O).

**Flujo actual a reemplazar:**
1. El usuario abre ARBA para ver la división parcelaria.
2. Mide manualmente superficies cubiertas en Google Maps/Earth.
3. Registra resultados a mano.

**Flujo objetivo:**
1. El usuario ingresa un polígono o coordenadas de la parcela.
2. La app descarga, procesa y segmenta imágenes satelitales automáticamente.
3. Exporta áreas en m², geometrías GeoJSON/Shapefile y visualización anotada.

---

## 1) Objetivos SMART

| # | Objetivo | Indicador | Plazo sugerido |
|---|----------|-----------|----------------|
| O1 | Detectar todas las superficies cubiertas (techos) en una parcela dada, sin intervención manual en el pipeline | IoU ≥ 0.75 vs medición humana | Sprint 3 |
| O2 | Calcular área real en m² con error ≤ 10% respecto a medición humana experta | MAPE ≤ 10% en dataset de validación Luján | Sprint 4 |
| O3 | Exportar resultados en CSV, GeoJSON y Shapefile utilizables en QGIS u otros SIG | Archivos válidos, abribles sin errores | Sprint 2 |
| O4 | Operar en Windows sin dependencias de pago, APIs privadas ni credenciales corporativas | Zero dependencias propietarias | Sprint 1 |
| O5 | Mejorar continuamente mediante corrección y reentrenamiento con datos locales de Luján | Ciclo de feedback implementado | Sprint 5–6 |

---

## 2) Alcance y Exclusiones

### ✅ Incluido
- Descarga automática de imágenes Sentinel-2 (ESA Copernicus Hub) y Landsat 8/9 (USGS EarthExplorer)
- Pipeline de procesamiento: corrección geométrica, cloud masking, normalización radiométrica
- Dos motores de detección intercambiables: clásico (índices espectrales + OpenCV) y deep learning (U-Net)
- Cálculo de área con conversión píxel → m² respetando CRS (EPSG:32720 / WGS84 UTM zona 20S)
- GUI de escritorio Windows (PyQt6/PySide6)
- Exportación: CSV, GeoJSON, Shapefile, imagen anotada (PNG)
- Módulo de validación: comparación manual vs automático, métricas IoU/F1
- Módulo de feedback y reentrenamiento incremental del modelo DL

### ❌ Excluido
- Acceso, consulta o comparación con bases de datos fiscales o catastrales (ARBA)
- Uso de Google Maps, Google Earth Engine o cualquier API propietaria en producción
- Análisis de altura de edificios o volumetría 3D
- Publicación web o acceso multiusuario simultáneo
- Análisis de series temporales (detección de cambios entre fechas)

### ⚠️ Limitación técnica crítica a documentar
Sentinel-2 tiene resolución nativa de 10 m/px. Edificios con superficie < ~50 m² son detectables pero con menor precisión. Para Luján (ciudad media, tejido residencial dominante con techos de ~60–300 m²) la resolución es adecuada. El sistema debe informar al usuario cuándo la resolución puede afectar la confiabilidad.

---

## 3) Requisitos Detallados

### 3.1 Funcionales (RF)

| ID | Requisito | Prioridad |
|----|-----------|-----------|
| RF-01 | Ingesta por polígono (GeoJSON/Shapefile externo), par de coordenadas o bbox dibujado en mapa | Alta |
| RF-02 | Descarga automática de Sentinel-2 L2A desde Copernicus Data Space Ecosystem (CDSE) sin credenciales de pago | Alta |
| RF-03 | Descarga alternativa manual: el usuario provee imagen ya descargada (formato GeoTIFF) | Alta |
| RF-04 | Cloud masking usando banda SCL de Sentinel-2 L2A | Alta |
| RF-05 | Motor clásico: segmentación por índices espectrales (NDVI inverso, NDBI) + morfología OpenCV | Media |
| RF-06 | Motor DL: U-Net preentrenado, con capacidad de fine-tuning local | Alta |
| RF-07 | Cálculo de área en m² por objeto detectado usando pyproj + resolución espacial del sensor | Alta |
| RF-08 | Exportación a CSV (área, coordenadas centroide), GeoJSON, Shapefile y PNG anotado | Alta |
| RF-09 | Módulo de corrección manual: el usuario puede editar polígonos detectados antes de exportar | Media |
| RF-10 | Módulo de feedback: el usuario valida resultados; las correcciones se almacenan para reentrenamiento | Media |
| RF-11 | Configuración de zona geográfica por defecto (Luján) con opción de cambio | Baja |

### 3.2 No Funcionales (RNF)

| ID | Requisito |
|----|-----------|
| RNF-01 | 100% open-source: licencias MIT, BSD, Apache 2.0, ODbL, CC-BY |
| RNF-02 | Compatible con Windows 10/11 (64-bit). Distribuible como ejecutable o entorno virtual |
| RNF-03 | Tiempo de procesamiento ≤ 3 min para una parcela típica (< 1 ha) en hardware sin GPU |
| RNF-04 | Código modular: cada componente (ingesta, procesamiento, detección, exportación) es independiente |
| RNF-05 | Documentación inline (docstrings Google style) en todas las funciones públicas |
| RNF-06 | Manejo explícito de errores con mensajes descriptivos al usuario (sin stack traces en GUI) |
| RNF-07 | Almacenamiento local: SQLite para metadatos, sistema de archivos para imágenes y modelos |

---

## 4) Casos de Uso / Escenarios

### CU-01: Análisis de parcela por coordenadas (caso principal)
**Actor:** Usuario técnico  
**Flujo:** Ingresa lat/lon o dibuja polígono en mapa → selecciona fecha → sistema descarga imagen → aplica pipeline → muestra resultado con área por objeto → usuario exporta CSV+GeoJSON.

### CU-02: Análisis con imagen propia (offline)
**Actor:** Usuario sin conexión o con imagen ya descargada  
**Flujo:** Carga GeoTIFF local → el sistema detecta resolución y CRS → ejecuta pipeline desde procesamiento → exporta resultados.

### CU-03: Corrección y feedback para reentrenamiento
**Actor:** Usuario experto validando resultados  
**Flujo:** Ejecuta detección → edita polígonos incorrectos en GUI → confirma → sistema guarda par (imagen-máscara corregida) en dataset local → permite lanzar fine-tuning.

### CU-04: Comparación manual vs automático (validación)
**Actor:** Usuario validando precisión del sistema  
**Flujo:** Ejecuta detección → introduce área medida manualmente → sistema calcula error % y lo registra en log de métricas.

---

## 5) Stakeholders y Roles

| Rol | Descripción | Responsabilidad en el proyecto |
|-----|-------------|-------------------------------|
| Fede (dev + domain expert) | Ingeniero civil, programador Python | Desarrollo, decisiones técnicas, validación de dominio |
| Compañeros de trabajo | Usuarios finales, no programadores | Pruebas de usabilidad, provisión de datos de validación (mediciones manuales) |
| Claude (agente IA) | Asistente de desarrollo | Generación de código, análisis de arquitectura, debugging |

---

## 6) Supuestos, Riesgos y Mitigaciones

| # | Supuesto / Riesgo | Probabilidad | Impacto | Mitigación |
|---|-------------------|-------------|---------|------------|
| S1 | Sentinel-2 cubre Luján con revisita de ~5 días, imágenes con < 20% nubosidad disponibles | Alta | — | Verificado: Luján está en zona de buena cobertura Copernicus |
| R1 | Resolución 10 m/px insuficiente para techos pequeños (< 30 m²) | Media | Alto | Documentar limitación; permitir input de imágenes de mayor resolución si el usuario las provee |
| R2 | API de Copernicus CDSE cambia o requiere registro | Baja | Alto | Implementar descarga manual como fallback; documentar proceso de registro (gratuito) |
| R3 | Modelo U-Net preentrenado tiene bajo rendimiento en techos de Luján (materiales locales distintos) | Media | Alto | Usar Google Open Buildings V3 (cubre ARG) como ground-truth para fine-tuning local; ciclo de feedback RF-10 |
| R4 | Distribución del ejecutable Windows con dependencias geoespaciales (GDAL, rasterio) compleja | Alta | Medio | Usar `conda-pack` o `PyInstaller` + GDAL wheels precompilados para Windows |
| R5 | Usuarios no técnicos cometen errores de entrada (CRS equivocado, polígono fuera de zona) | Media | Bajo | Validación de inputs con mensajes claros; CRS detectado automáticamente |

---

## 7) Entregables y Cronograma Sugerido

| Sprint | Duración | Entregable |
|--------|----------|------------|
| S1 | 1–2 semanas | Módulo de ingesta: descarga Sentinel-2 automática + manual. Tests de conectividad |
| S2 | 1–2 semanas | Pipeline de preprocesamiento: cloud masking, reproyección, normalización. Exportación básica |
| S3 | 2 semanas | Motor clásico (índices + morfología). GUI mínima funcional (PyQt6) |
| S4 | 2–3 semanas | Motor DL: U-Net con pesos preentrenados en datos abiertos (SpaceNet, Inria, o fine-tune con Open Buildings ARG) |
| S5 | 1–2 semanas | Módulo de validación + feedback + ciclo de reentrenamiento |
| S6 | 1 semana | Empaquetado Windows, documentación de usuario, pruebas con compañeros |

**Total estimado:** 10–14 semanas (trabajo parcial)

---

## 8) Métricas de Éxito y KPI

| Métrica | Fórmula | Umbral mínimo | Umbral objetivo |
|---------|---------|---------------|-----------------|
| IoU (Intersection over Union) | `|pred ∩ gt| / |pred ∪ gt|` | ≥ 0.65 | ≥ 0.80 |
| Error de área (MAPE) | `mean(|area_pred - area_manual| / area_manual) × 100` | ≤ 15% | ≤ 8% |
| Precisión (Precision) | `TP / (TP + FP)` | ≥ 0.75 | ≥ 0.85 |
| Recall | `TP / (TP + FN)` | ≥ 0.70 | ≥ 0.80 |
| Tiempo de procesamiento | Segundos por parcela (< 1 ha, sin GPU) | ≤ 300 s | ≤ 120 s |

**Dataset de validación:** Mínimo 20 parcelas de Luján medidas manualmente por usuarios.  
**Baseline:** Medición humana en Google Maps/Earth (referencia de verdad de campo).

---

## 9) Recomendaciones de Diseño y Tecnología

### 9.1 Arquitectura recomendada

```
roofscan/
├── core/
│   ├── ingesta/          # downloader.py (Sentinel/Landsat) + loader.py (GeoTIFF local)
│   ├── preproceso/       # cloud_mask.py, normalizer.py, reprojector.py
│   ├── deteccion/
│   │   ├── clasico/      # spectral_indices.py, morphology.py
│   │   └── dl/           # unet.py, trainer.py, predictor.py
│   ├── calculo/          # area_calculator.py, geometry_merger.py
│   └── exportacion/      # csv_exporter.py, geojson_exporter.py, shp_exporter.py
├── gui/                  # PyQt6: main_window.py, map_widget.py, results_panel.py
├── data/
│   ├── models/           # pesos .pth del modelo U-Net
│   ├── feedback/         # dataset local acumulado (imágenes + máscaras)
│   └── cache/            # imágenes satelitales descargadas
├── tests/
└── docs/
```

### 9.2 Stack tecnológico validado

| Componente | Librería | Justificación |
|------------|----------|---------------|
| GUI | PyQt6 / PySide6 | Stack habitual del dev, maduro en Windows |
| Descarga satelital | `sentinelsat` o `cdsetool` | Acceso a CDSE (reemplaza Copernicus SciHub) |
| Raster I/O | `rasterio` + `GDAL` | Estándar de facto GIS Python |
| Geometrías | `shapely`, `geopandas`, `pyproj` | Operaciones vectoriales + reproyección |
| CV clásico | `OpenCV`, `scikit-image`, `numpy` | Morphología, umbralización, índices |
| Deep Learning | `PyTorch` + `segmentation-models-pytorch` | U-Net, ResNet encoder, GPU/CPU compatible |
| Visualización | `matplotlib`, `folium` (mapas interactivos) | Overlay + HTML maps |
| Datos | `pandas`, `SQLite` (via `sqlite3`) | Metadatos y registro de feedback |
| Empaquetado | `PyInstaller` + GDAL wheels | Distribución Windows sin instalación |

### 9.3 Ground-truth para entrenamiento DL

- **Google Open Buildings V3** (CC BY-4.0): cubre Argentina, ~1.8B polígonos de edificios derivados de imágenes 50 cm. Descargable por país desde `source.coop`. **Usar como dataset de validación y fine-tuning local.**
- **Microsoft Global ML Building Footprints** (ODbL): cobertura complementaria de América del Sur.
- **OpenStreetMap** vía Overpass API o Geofabrik: buildings en Luján (cobertura variable, verificar antes de usar).
- **Combinado VIDA** (Google+Microsoft+OSM, ODbL): 2.7B footprints, particionado por país ISO.

### 9.4 Modelo U-Net: estrategia de preentrenamiento

1. **Usar `segmentation-models-pytorch`** con encoder ResNet34 preentrenado en ImageNet.
2. **Fine-tuning** con pares (recorte Sentinel-2 Luján / máscara de Open Buildings ARG filtrada por confianza > 0.7).
3. **Feedback loop:** cada corrección manual del usuario se acumula en `data/feedback/`; reentrenamiento periódico con `trainer.py`.

---

## 10) Referencias a Buenas Prácticas y Justificativos

| Decisión | Justificativo |
|----------|---------------|
| Sentinel-2 como fuente primaria | 10 m/px es adecuado para techos residenciales en Luján (media 80–300 m²); revisita 5 días; completamente gratuito vía CDSE |
| U-Net como arquitectura DL | Arquitectura estándar para segmentación semántica en imágenes satelitales; amplia literatura de aplicación a edificios; disponible en `segmentation-models-pytorch` |
| Google Open Buildings V3 como ground-truth | Cubre Argentina con licencia CC BY-4.0; derivado de imágenes 50 cm; incluye score de confianza por polígono |
| Motor clásico como fallback | Garantiza funcionamiento sin modelo entrenado; útil para zonas con techos espectralmente distintos (zinc, tejas) |
| PyQt6 para GUI | Consistencia con stack del desarrollador; distribuible en Windows; soporte para widgets de mapa via `folium` embedded |
| EPSG:32720 como CRS de trabajo | UTM zona 20S cubre Luján correctamente; unidades métricas nativas para cálculo de área preciso |

**Recursos clave:**
- Copernicus Data Space Ecosystem: https://dataspace.copernicus.eu
- Google Open Buildings V3: https://sites.research.google/gr/open-buildings/
- VIDA combined dataset (Google+MS+OSM): https://source.coop/vida/google-microsoft-osm-open-buildings
- `segmentation-models-pytorch`: https://github.com/qubvel/segmentation_models.pytorch
- Microsoft Global ML Footprints: https://github.com/microsoft/GlobalMLBuildingFootprints
- Geofabrik Argentina (OSM): https://download.geofabrik.de/south-america/argentina.html

---

*Este documento es el contexto de largo plazo del proyecto RoofScan. Actualizarlo en cada sprint con decisiones tomadas, librerías descartadas y métricas obtenidas.*
