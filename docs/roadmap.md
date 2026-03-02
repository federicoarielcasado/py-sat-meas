# RoofScan — Roadmap

Estado al 2026-03-02. Sprints 1–6 completados.

---

## ✅ Implementado

| RF | Descripción |
|----|-------------|
| RF-01 (parcial) | Ingesta por coordenadas / GeoTIFF local. Falta: dibujar polígono/bbox en mapa. |
| RF-02 (parcial) | `downloader.py` funcional vía CDSE. Falta: integración en GUI. |
| RF-03 | Carga manual de GeoTIFF local. |
| RF-04 | Cloud masking con banda SCL (archivo separado cargable en GUI). |
| RF-05 | Motor clásico: NDVI / NDBI / NDWI + morfología. |
| RF-06 | Motor U-Net: ResNet34 encoder, tiling + Gaussian blending, fine-tuning local. |
| RF-07 | Cálculo de área en m² con EPSG:32720. |
| RF-08 | Exportación CSV, GeoJSON, Shapefile, PNG anotado. |
| RF-09 | ❌ Edición manual de polígonos. Baja prioridad, diferido. |
| RF-10 | Feedback + log de validaciones + reentrenamiento incremental U-Net. |
| RF-11 | Zona por defecto: Luján (configurable en `config.py`). |

---

## 🔶 Pendiente

### 1. Integrar descarga Sentinel-2 en GUI *(alta prioridad)*

`downloader.py` ya existe pero la ventana principal no lo expone. El usuario
debe descargar manualmente y cargar un GeoTIFF.

**Trabajo a realizar:**
- Panel o diálogo con campos: bbox (del mapa o manual), fecha inicio/fin, nubosidad máxima.
- `DownloadWorker(QObject)` en QThread, análogo a `AnalysisWorker`.
- Barra de progreso con estado de la descarga.
- Al terminar, auto-cargar la imagen descargada en el flujo principal.

---

### 2. Pesos U-Net base pre-entrenados *(alta prioridad)*

Sin pesos `data/models/unet_best.pth`, el motor U-Net es inutilizable hasta
que el usuario acumule ≥ 2 pares de feedback y entrene localmente.

**Estrategia:**
1. Descargar Google Open Buildings V3 para Argentina desde `source.coop`.
2. Filtrar polígonos con `confidence > 0.7`.
3. Recortar pares (imagen Sentinel-2 / máscara rasterizada) en zona de Luján.
4. Entrenar fuera de la GUI con un script dedicado:
   `python scripts/pretrain_unet.py --tiles-dir data/pretrain/ --epochs 50`
5. Distribuir `unet_best.pth` junto con el ejecutable.

---

### 3. Bbox/polígono dibujable en el mapa *(media prioridad)*

`MapWidget` detecta clics y emite `geo_clicked(lat, lon)`, pero no permite
trazar geometrías interactivas para definir el área de análisis.

**Opciones de implementación:**
- **Rubber-band Qt:** dibujar rectángulo con `QRubberBand` sobre el widget de imagen.
- **Folium editable:** usar `folium.plugins.Draw` en el HTML embebido + bridge JS→Python.

---

### 4. Exportadores modulares GeoJSON / Shapefile *(media prioridad)*

La exportación de vectores está inline en `main_window.py`. Extraer a módulos
dedicados para consistencia arquitectónica y facilitar tests unitarios:

- `roofscan/core/exportacion/geojson_exporter.py` → `export_geojson(gdf, output_dir)`
- `roofscan/core/exportacion/shp_exporter.py` → `export_shapefile(gdf, output_dir)`

---

### 5. Validación con datos reales de Luján *(necesario para certificar KPIs)*

Las métricas (IoU, MAPE) están implementadas pero no se han corrido contra
parcelas reales. Para certificar los objetivos O1 y O2:

- Medir manualmente ≥ 20 parcelas en Luján.
- Correr la detección automática sobre las mismas.
- Usar el módulo de validación (pestaña *Validación* en la GUI) para registrar resultados.
- Objetivo: **IoU ≥ 0.75**, **MAPE ≤ 10%**.

---

### 6. RF-09: Edición manual de polígonos *(baja prioridad)*

Permitir al usuario ajustar polígonos detectados antes de exportar.
Requiere una capa vectorial editable superpuesta al mapa.
Diferido indefinidamente por complejidad de implementación.
