# RoofScan — Roadmap

Estado al 2026-03-03. Sprints 1–6 completados.

---

## ✅ Implementado

| RF | Descripción |
|----|-------------|
| RF-01 (parcial) | Ingesta por coordenadas / GeoTIFF local. Falta: dibujar polígono/bbox en mapa. |
| RF-02 ✅ | Descarga Sentinel-2 integrada en GUI: `DownloadDialog` con búsqueda, tabla de escenas y auto-selección por menor nubosidad. Auto-carga al terminar. |
| RF-03 | Carga manual de GeoTIFF local. |
| RF-04 | Cloud masking con banda SCL (archivo separado cargable en GUI). |
| RF-05 | Motor clásico: NDVI / NDBI / NDWI + morfología. |
| RF-06 | Motor U-Net: ResNet34 encoder, tiling + Gaussian blending, fine-tuning local. |
| RF-07 | Cálculo de área en m² con EPSG:32720. |
| RF-08 | Exportación CSV, GeoJSON, Shapefile, PNG anotado. |
| RF-09 | ❌ Edición manual de polígonos. Baja prioridad, diferido. |
| RF-10 | Feedback + log de validaciones + reentrenamiento incremental U-Net. |
| RF-11 | Zona por defecto: Luján (configurable en `config.py`). |
| RF-12 ✅ | Conversión .SAFE → GeoTIFF multibanda (`safe_loader.py`): apila B02-B12, remuestrea 20m→10m, exporta SCL separado. |
| RF-13 ✅ | Scripts de preentrenamiento U-Net: `prepare_tiles.py` (tiles desde Open Buildings + Sentinel-2) y `pretrain_unet.py` (entrena con BCE+Dice loss). |
| RF-14 ✅ | Mensura masiva por parcela: `wfs_arba.py` (3 estrategias de obtención), `spatial_join.py` (intersección techo↔parcela) y `batch_mensura.py` (pipeline CLI completo). |

---

## 🔶 Pendiente

### 1. ~~Integrar descarga Sentinel-2 en GUI~~ *(completado)*

---

### 2. ~~Pesos U-Net base pre-entrenados~~ *(completado)*

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

---

### 7. Mensura masiva del partido de Luján *(objetivo principal)*

**Objetivo:** procesar automáticamente la totalidad de las parcelas del partido
de Luján y obtener el área de techo de cada una, sin intervención manual.

**Infraestructura ya implementada (RF-14):**
- `wfs_arba.py`: obtiene polígonos de parcelas con 3 estrategias en cascada:
  1. Archivo catastral local (`.gpkg` / `.shp`) — más rápido y confiable.
  2. WFS automático (prueba múltiples endpoints de ARBA e IDERA).
  3. Escaneo WMS GetFeatureInfo en grilla — fallback lento pero garantizado.
- `spatial_join.py`: intersección exacta techo ↔ parcela con área, % cubierto y n_techos.
- `scripts/batch_mensura.py`: pipeline CLI end-to-end.

**Bloqueante principal — datos catastrales:**
El WFS de ARBA (`geo.arba.gov.ar`) está deshabilitado en el servidor actual.
Para el partido completo se requiere una de las siguientes fuentes:

| Fuente | URL | Formato |
|--------|-----|---------|
| Datos Abiertos Buenos Aires *(recomendada)* | https://datos.gba.gob.ar/dataset/catastro-territorial | GeoPackage / SHP |
| IGN — Capas SIG Argentina | https://www.ign.gob.ar/NuestrasActividades/InformacionGeoespacial/CapasIG | SHP |

Descarga única → guardar en `data/catastro/lujan.gpkg` → el script la reutiliza en adelante.

**Flujo completo una vez con el catastro disponible:**
```bash
# 1. Descargar imagen Sentinel-2 de Luján (desde la GUI o CLI)
# 2. Ejecutar mensura
python scripts/batch_mensura.py \
    --image data/cache/S2A_MSIL2A_..._stacked.tif \
    --parcelas data/catastro/lujan.gpkg \
    --output data/output/mensura_lujan.csv \
    --output-geojson
```

**Salida:** CSV + GeoJSON con una fila por parcela:
`nomenclatura, partido, seccion, manzana, area_parcela_m2, area_techos_m2, n_techos, pct_cubierto`

**Flujo alternativo — lista de partidas:**
Si se dispone de un listado de códigos de nomenclatura catastral:
```bash
python scripts/batch_mensura.py \
    --image data/cache/S2A_..._stacked.tif \
    --parcelas data/catastro/lujan.gpkg \
    --partidas data/mis_partidas.csv \
    --output data/output/mensura_parcelas.csv
```

**Pendiente para completar el objetivo:**
- [ ] Descargar y validar el archivo catastral del partido de Luján.
- [ ] Ejecutar el pipeline sobre una imagen real y verificar resultados.
- [ ] Si la detección clásica no es suficientemente precisa, correr con motor U-Net (requiere preentrenamiento previo con `prepare_tiles.py` + `pretrain_unet.py`).
