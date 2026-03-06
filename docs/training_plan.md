# Plan de Entrenamiento — RoofScan

**Fecha:** 2026-03-06

Guía para mejorar la precisión de detección y mensura de techos a través
del ciclo de entrenamiento incremental del modelo U-Net.

---

## Resumen de precisión esperada

| Nivel | Esfuerzo | IoU esperado | MAPE área |
|-------|----------|-------------|-----------|
| Motor Clásico (sin entreno) | 0 h | ~0.50 | ~20% |
| U-Net fine-tune con 10 pares locales | 2–3 h | ~0.65–0.70 | ~12% |
| U-Net preentrenado con Open Buildings | 1 día | ~0.75–0.80 | ~8% |

**Objetivo KPI del proyecto:** IoU ≥ 0.75 · MAPE ≤ 10% (ver `CLAUDE.md`)

---

## Nivel 1 — Feedback loop desde la GUI (sin datos externos)

El camino más rápido para obtener un modelo adaptado a Luján.
No requiere descargar datasets externos ni GPU.

### Pasos

1. **Abrir la aplicación:**
   ```
   python -m roofscan
   ```

2. **Cargar una imagen Sentinel-2** de Luján (`.tif` con 6 bandas apiladas).

3. **Detectar techos** con el motor Clásico (ajustar umbrales NDVI/NDBI/NDWI
   en el panel de parámetros hasta obtener un resultado razonable visualmente).

4. **Registrar feedback** en la pestaña Validación:
   - Ingresar el área real de techos (medida en Google Maps).
   - Guardar el par imagen-máscara como feedback.

5. **Repetir** los pasos 2–4 con 5 a 10 imágenes de zonas distintas de Luján.

6. **Reentrenar** haciendo clic en el botón "Reentrenar modelo" (pestaña
   Validación). El proceso corre en background (~10 min sin GPU, 20 épocas).

7. Los pesos se guardan en `data/models/unet_best.pth`. Cambiar el motor
   a U-Net para usar el modelo entrenado en la próxima detección.

### Notas
- Cada ciclo de feedback mejora el modelo; el entrenamiento es incremental
  (fine-tune sobre los pesos previos, no desde cero).
- Con menos de 2 pares de feedback el botón de reentrenamiento estará
  deshabilitado.

---

## Nivel 2 — Preentrenamiento con Open Buildings (recomendado)

Produce un punto de partida mucho mejor que ImageNet puro, especialmente
para techos de tejido residencial denso como Luján.

### Requisitos
- Python con PyTorch instalado (`pip install torch torchvision`)
- Imagen Sentinel-2 de Luján en formato GeoTIFF multibanda (6 bandas)
- Polígonos de edificios para Argentina (ver opciones abajo)
- ~4 GB de espacio en disco

### Paso 1 — Descargar polígonos de edificios para Argentina

**Opción A (recomendada) — VIDA combined dataset** (Google + MS + OSM, ODbL):
```
https://source.coop/vida/google-microsoft-osm-open-buildings
```
Descargar `ARG.gpkg` (~700 MB). Guardar en `data/pretrain/ARG.gpkg`.

**Opción B — Google Open Buildings V3** (CC BY-4.0):
```
https://sites.research.google/gr/open-buildings/
```
Filtrar por Argentina y descargar el GeoPackage.

**Opción C — OpenStreetMap via Geofabrik** (ODbL):
```
https://download.geofabrik.de/south-america/argentina.html
```
Extraer la capa `buildings` con ogr2ogr del `.osm.pbf`.

### Paso 2 — Generar tiles de entrenamiento

```bash
python scripts/prepare_tiles.py \
    --buildings data/pretrain/ARG.gpkg \
    --tiles-dir data/cache/ \
    --output-dir data/pretrain/ \
    --confidence 0.7 \
    --tile-size 256 \
    --stride 128 \
    --min-roof-pct 1.0
```

- `--tiles-dir`: directorio con los GeoTIFFs Sentinel-2 de referencia.
- `--confidence 0.7`: filtra polígonos de baja confianza (solo para VIDA/Google).
- `--stride 128`: paso de la ventana deslizante (128 < tile-size → solapamiento 50%).
- Duración: 5–15 min dependiendo del área cubierta.

El resultado se guarda en:
```
data/pretrain/
    images/   # float32 (6, 256, 256) normalizado [0, 1]
    masks/    # uint8   (256, 256), 1=techo 0=fondo
```

### Paso 3 — Entrenar el modelo

```bash
# Sin GPU — lento pero funciona (~2-4 h para 500 tiles, 50 épocas)
python scripts/pretrain_unet.py \
    --tiles-dir data/pretrain/ \
    --output data/models/unet_best.pth \
    --epochs 50 \
    --lr 1e-4

# Con GPU NVIDIA — 10–20× más rápido
python scripts/pretrain_unet.py \
    --tiles-dir data/pretrain/ \
    --output data/models/unet_best.pth \
    --epochs 80 \
    --lr 5e-5 \
    --batch-size 8
```

Parámetros recomendados según tamaño del dataset:

| Tiles generados | Épocas | LR    |
|-----------------|--------|-------|
| < 200           | 30     | 1e-4  |
| 200–1000        | 50     | 1e-4  |
| > 1000          | 80     | 5e-5  |

### Paso 4 — Continuar con feedback loop (Nivel 1)

Una vez obtenidos los pesos preentrenados, continuar con el ciclo de
feedback desde la GUI para adaptar el modelo a las condiciones específicas
de Luján (materiales, sombras, orientación de techos).

---

## Nivel 3 — Clasificador de materiales (trabajo futuro)

El módulo `material_classifier.py` implementa un clasificador en cascada:
CNN multi-escala + MLP espectral → reglas espectrales como fallback.

Actualmente funciona solo con reglas espectrales (sin entrenamiento).
Para entrenar el MLP/CNN se necesitan **etiquetas de material por techo**.

### Proceso de etiquetado

1. Exportar el GeoJSON de techos detectados desde la GUI.
2. Abrir en QGIS.
3. Agregar columna `material` con valores:
   - `zinc_corrugado`
   - `losa_hormigon`
   - `tejas_ceramica`
   - `construccion_incompleta`
4. Inspeccionar visualmente en Google Maps satelital para cada techo.
5. Guardar como `data/pretrain/lujan_techos_etiquetados.geojson`.

### Entrenamiento (script a desarrollar)

Una vez disponibles ~200 techos etiquetados por zona, crear
`scripts/train_material_classifier.py` que:
- Llame a `extract_spectral_stats()` para generar features por techo.
- Entrene el MLP con `build_material_mlp()`.
- Guarde los pesos en `data/models/material_mlp.pth`.

Usar con el script de mensura masiva:
```bash
python scripts/batch_mensura.py \
    --image data/cache/lujan_stacked.tif \
    --parcelas data/catastro/lujan_parcelas.gpkg \
    --output data/output/mensura_lujan.csv \
    --material \
    --material-mlp data/models/material_mlp.pth \
    --output-geojson \
    --classify
```

---

## Validación de KPIs

Para verificar si el modelo cumple los objetivos del proyecto, medir
al menos 20 parcelas de Luján manualmente en Google Maps y comparar:

```bash
# Correr mensura sobre el conjunto de validación
python scripts/batch_mensura.py \
    --image data/cache/lujan_stacked.tif \
    --parcelas data/validacion/parcelas_muestra.gpkg \
    --output data/output/validacion_resultado.csv \
    --output-geojson

# Comparar con mediciones manuales usando la pestaña Validación de la GUI
# o directamente en Python:
python -c "
import pandas as pd
pred = pd.read_csv('data/output/validacion_resultado.csv')
manual = pd.read_csv('data/validacion/mediciones_manuales.csv')
mape = ((pred['area_techos_m2'] - manual['area_manual_m2']).abs() / manual['area_manual_m2']).mean() * 100
print(f'MAPE: {mape:.1f}%  (objetivo: ≤ 10%)')
"
```

---

## Flujo recomendado para primera mensura completa de Luján

```
1. python scripts/download_catastro.py
   → data/catastro/lujan_parcelas.gpkg  (~140k parcelas, 5-8 min)

2. Descargar imagen Sentinel-2 desde la GUI (⬇ Descargar Sentinel-2…)
   → data/cache/lujan_YYYYMMDD_stacked.tif

3. [Opcional] Entrenar U-Net con Open Buildings (Nivel 2 arriba)

4. python scripts/batch_mensura.py \
       --image data/cache/lujan_stacked.tif \
       --parcelas data/catastro/lujan_parcelas.gpkg \
       --output data/output/mensura_lujan.csv \
       --output-geojson \
       --classify
   → data/output/mensura_lujan.csv
   → data/output/mensura_lujan_roofs.geojson

5. Abrir mensura_lujan.csv en Excel / QGIS para revisar resultados.
```
