# RoofScan — Guía de Usuario

**Versión 1.0** | Detección automática de superficies cubiertas con imágenes satelitales

---

## Índice

1. [Requisitos e instalación](#1-requisitos-e-instalación)
2. [Primer inicio](#2-primer-inicio)
3. [Buscar una dirección y ver parcelas ARBA](#3-buscar-una-dirección-y-ver-parcelas-arba)
4. [Cargar una imagen satelital GeoTIFF](#4-cargar-una-imagen-satelital-geotiff)
5. [Cómo obtener imágenes Sentinel-2](#5-cómo-obtener-imágenes-sentinel-2)
6. [Ejecutar la detección de techos](#6-ejecutar-la-detección-de-techos)
7. [Motor Clásico vs U-Net](#7-motor-clásico-vs-u-net)
8. [Interpretar los resultados](#8-interpretar-los-resultados)
9. [Exportar resultados](#9-exportar-resultados)
10. [Validación y mejora continua](#10-validación-y-mejora-continua)
11. [Limitaciones técnicas](#11-limitaciones-técnicas)
12. [Solución de problemas frecuentes](#12-solución-de-problemas-frecuentes)

---

## 1. Requisitos e instalación

### Instalación con Python (entorno virtual)

**Requisitos mínimos:**
- Windows 10 o 11 (64-bit)
- Python 3.11 o superior
- 4 GB de RAM mínimo (8 GB recomendado)
- ~2 GB de espacio en disco

**Pasos:**

1. Descomprimí el archivo del proyecto en una carpeta de trabajo (ej. `C:\roofscan\`).

2. Abrí una terminal (CMD o PowerShell) en esa carpeta y ejecutá:
   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   > **Nota:** La instalación puede tardar varios minutos dependiendo de la velocidad de internet. GDAL y rasterio son paquetes grandes.

3. Copiá el archivo de configuración:
   ```bat
   copy .env.example .env
   ```

4. Abrí `.env` con el Bloc de notas y completá tus credenciales de Copernicus Data Space (ver sección 5).

5. Lanzá la aplicación:
   ```bat
   python -m roofscan
   ```

### Instalación con ejecutable (si se provee)

Si recibiste un archivo `RoofScan.exe`, simplemente ejecutalo. No requiere instalación de Python.

---

## 2. Primer inicio

Al abrir RoofScan verás tres zonas principales:

```
┌─────────────────┬──────────────────────────────────────────┐
│ Panel izquierdo │         Vista central (mapa/imagen)       │
│                 │                                           │
│ • Localización  │   [Imagen / Detección] [Resultados]       │
│ • Imagen        │               [Validación]                │
│ • Detección     │                                           │
│ • Motor         │                                           │
│ • ▶ Detectar   │                                           │
│ • Exportar      │                                           │
└─────────────────┴──────────────────────────────────────────┘
│                      Barra de estado                        │
└─────────────────────────────────────────────────────────────┘
```

La **barra de estado** (abajo) muestra el progreso de todas las operaciones.

---

## 3. Buscar una dirección y ver parcelas ARBA

### Buscar por dirección

1. En el panel **Localización**, escribí la dirección en el campo de texto.
   - Ejemplo: `San Martín 456, Luján, Buenos Aires`
   - La búsqueda usa OpenStreetMap (requiere conexión a internet).

2. Presioná **🔍 Buscar** o la tecla Enter.

3. El mapa mostrará la zona centrada en esa dirección con el overlay de parcelas ARBA activado automáticamente.

### Overlay de parcelas ARBA

El **overlay de parcelas** muestra los límites catastrales de la zona sobre la imagen. Sirve para identificar con precisión qué parcela analizar.

- Activar/desactivar: checkbox **"Mostrar parcelas ARBA"**.
- Requiere conexión a internet (consulta el servidor WMS de IDERA/ARBA).

### Seleccionar una parcela

1. Con el overlay activo, hacé **clic sobre una parcela** en el mapa.
2. El panel mostrará información catastral: nomenclatura, partido, sección, manzana, parcela.
3. Presioná **"Usar esta parcela como área de análisis"** para registrar el área de interés.

> **Tip:** El botón "Usar esta parcela" no descarga imágenes automáticamente (ver sección 5 para la descarga). Sirve para identificar el bbox de la parcela que necesitás.

---

## 4. Cargar una imagen satelital GeoTIFF

Una vez que tenés la imagen satelital (ver sección 5), cargala en la app:

1. En el panel **Imagen satelital**, hacé clic en **"Abrir GeoTIFF…"**
2. Seleccioná el archivo `.tif` con las bandas espectrales.
3. La imagen aparecerá en el mapa en color verdadero (RGB).

### ¿Cuántas bandas necesito?

RoofScan usa 6 bandas de Sentinel-2 en este orden:

| Posición | Banda S2 | Descripción |
|----------|----------|-------------|
| 1 | B02 | Azul |
| 2 | B03 | Verde |
| 3 | B04 | Rojo |
| 4 | B08 | Infrarrojo cercano (NIR) |
| 5 | B11 | SWIR-1 |
| 6 | B12 | SWIR-2 |

El GeoTIFF debe tener estas 6 bandas apiladas en ese orden. Si usás el script de descarga incluido, el archivo ya viene en el formato correcto.

### Cargar SCL (máscara de nubes, opcional pero recomendado)

El archivo SCL (Scene Classification Layer) permite enmascarar automáticamente píxeles nubosos antes de la detección.

1. Hacé clic en **"Cargar SCL…"**
2. Seleccioná el archivo `*_SCL_20m.tif` (viene junto con la escena Sentinel-2).
3. Un indicador verde confirmará que el SCL está cargado.
4. Para quitarlo, presioná el botón **✕**.

> **Sin SCL:** La detección funciona igualmente, pero píxeles nubosos pueden generar falsos positivos (nubes detectadas como techos).

---

## 5. Cómo obtener imágenes Sentinel-2

### Opción A: Descarga automática (requiere registro gratuito)

1. Registrate gratis en [Copernicus Data Space Ecosystem (CDSE)](https://dataspace.copernicus.eu).
2. Completá tus credenciales en el archivo `.env`:
   ```
   CDSE_USER=tu_email@ejemplo.com
   CDSE_PASSWORD=tu_contraseña
   ```
3. Usá el script de descarga desde la terminal:
   ```python
   from roofscan.core.ingesta.downloader import download_sentinel2
   from pathlib import Path

   paths = download_sentinel2(
       bbox=(-59.12, -34.58, -59.08, -34.55),   # lon_min, lat_min, lon_max, lat_max
       date_range=("2024-01-01", "2024-03-31"),
       output_dir=Path("data/cache/"),
       max_cloud_pct=20,
   )
   print(f"Descargadas: {len(paths)} escenas")
   ```

### Opción B: Descarga manual desde el portal web

1. Entrá a [browser.dataspace.copernicus.eu](https://browser.dataspace.copernicus.eu)
2. Iniciá sesión con tu cuenta CDSE.
3. Buscá la zona de interés (Luján, Buenos Aires, Argentina).
4. Filtrá por:
   - Satélite: **Sentinel-2**
   - Nivel: **L2A** (con corrección atmosférica)
   - Nubosidad: **< 20%**
5. Descargá la escena y extraé el archivo `.SAFE`.
6. Apilá las bandas B02, B03, B04, B08, B11, B12 en un GeoTIFF (con QGIS, SNAP o gdal_merge.py).

### ¿Qué período usar?

Para Luján, las mejores condiciones (menor nubosidad) son:
- **Diciembre a Marzo**: alta probabilidad de nubes pero también de sol entre tormentas.
- **Junio a Agosto**: cielos más despejados, ideal para imágenes de calidad.

Sentinel-2 tiene revisita de **~5 días** sobre Luján.

---

## 6. Ejecutar la detección de techos

1. Con la imagen cargada, configurá los **parámetros de detección** (o dejá los valores por defecto).
2. Seleccioná el **motor de detección** (ver sección 7).
3. Presioná **"▶ Detectar techos"**.
4. La barra de estado mostrará el progreso: carga → preprocesamiento → detección → áreas.
5. Al terminar, la detección aparece en el mapa y la tabla de resultados se completa.

### Parámetros del motor Clásico

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| NDVI máx | 0.20 | Umbral de vegetación. Aumentar si se detectan árboles como techos. |
| NDBI mín | -0.05 | Umbral de construido. Disminuir si faltan techos oscuros. |
| NDWI máx | 0.05 | Umbral de agua. Aumentar si se detectan ríos como techos. |
| Área mín (px) | 5 | Ignorar objetos menores a N píxeles (~500 m² en S2). |

---

## 7. Motor Clásico vs U-Net

### Motor Clásico (índices espectrales)

**Cuándo usarlo:**
- Siempre disponible, sin necesidad de entrenamiento previo.
- Útil para comenzar a explorar una zona nueva.
- Cuando las superficies tienen distintas firmas espectrales claras (zinc, fibrocemento, losa).

**Cómo funciona:**
Calcula índices espectrales (NDVI, NDBI, NDWI) por píxel y aplica umbrales para identificar superficies construidas. Luego aplica morfología matemática para limpiar la máscara y separar objetos.

**Limitaciones:**
- Puede confundir superficies no vegetadas (suelo desnudo, arena) con techos.
- Sensible a la fecha de la imagen (NDVI varía con las estaciones).

### Motor U-Net (Deep Learning)

**Cuándo usarlo:**
- Después de haber acumulado feedback y reentrenado el modelo (ver sección 10).
- Generalmente más preciso en zonas con variedad de materiales de techo.
- Cuando el motor clásico produce muchos falsos positivos o negativos.

**Requisito:** Necesita el archivo `data/models/unet_best.pth` generado por el reentrenamiento. Si no existe, el sistema muestra un mensaje explicando cómo generarlo.

**Las vistas de índice (NDVI, NDBI, NDWI) no están disponibles con U-Net**, ya que este motor no las calcula.

---

## 8. Interpretar los resultados

### Pestaña "Imagen / Detección"

Selector de vista:
- **RGB True Color**: imagen en color natural (bandas Rojo, Verde, Azul).
- **Overlay detección**: techos detectados en naranja sobre la imagen RGB.
- **NDVI**: índice de vegetación (verde = vegetación, rojo = no vegetación). *Solo motor clásico.*
- **NDBI**: índice de superficies construidas. *Solo motor clásico.*
- **NDWI**: índice de agua. *Solo motor clásico.*

### Pestaña "Resultados"

Tabla con una fila por objeto detectado:
- **ID**: identificador numérico.
- **Área (m²)**: área real calculada en metros cuadrados.
- **Área (px)**: área en píxeles.
- **Centroide**: posición del centroide en píxeles (fila, columna).

El resumen muestra el **total de techos detectados** y el **área total cubierta**.

### ¿Cómo entender el área calculada?

El área se calcula como: `área_px × (resolución_m)²`

Para Sentinel-2: `área_px × 10² = área_px × 100 m²`

Ejemplo: un techo de 10 píxeles = ~1.000 m². La resolución de 10 m/px limita la precisión para objetos pequeños.

---

## 9. Exportar resultados

Todos los formatos se exportan desde el panel **"Exportar resultados"** (habilitados tras la detección):

| Botón | Formato | Uso |
|-------|---------|-----|
| Guardar GeoTIFF procesado | `.tif` | Imagen preprocesada (normalizada, sin nubes) |
| Guardar PNG de previsualización | `.png` | Imagen RGB lista para compartir o informes |
| Guardar GeoJSON de techos | `.geojson` | Polígonos vectoriales, abribles en QGIS, Google Maps |
| Guardar Shapefile de techos | `.shp` | Formato estándar GIS, compatible con ArcGIS y QGIS |
| Guardar CSV de áreas | `.csv` | Planilla con id, área (m²), área (px), centroide proyectado y WGS84 |

### Usar el GeoJSON/Shapefile en QGIS

1. Abrí QGIS.
2. Arrastrá el archivo `.geojson` o `.shp` al panel de capas.
3. Los polígonos aparecerán con los atributos `id`, `area_m2`, `area_px`.
4. Podés estilizarlos, calcular estadísticas y combinarlos con capas catastrales.

### Columnas del CSV

| Columna | Descripción |
|---------|-------------|
| `id` | Identificador del objeto |
| `area_m2` | Área en metros cuadrados |
| `area_px` | Área en píxeles |
| `centroid_row/col` | Centroide en coordenadas de píxel |
| `centroid_x/y_m` | Centroide en sistema proyectado (UTM 20S, metros) |
| `centroid_lon/lat` | Centroide en WGS84 (grados decimales) |

---

## 10. Validación y mejora continua

La pestaña **"Validación"** permite mejorar la precisión del modelo U-Net con el tiempo.

### Calcular el error de área

1. Después de ejecutar una detección, andá a la pestaña **Validación**.
2. El campo **"Área automática"** muestra el total detectado.
3. Ingresá el **área medida manualmente** (en m²) en el campo correspondiente.
4. Presioná **"Calcular error"**.
5. El sistema muestra el MAPE (error porcentual):
   - 🟢 Verde: ≤ 10% (excelente)
   - 🟡 Amarillo: ≤ 20% (aceptable)
   - 🔴 Rojo: > 20% (mejorar con más feedback)

El resultado queda registrado en el historial (`data/feedback/validation_log.csv`).

### Guardar feedback para reentrenamiento

1. Después de una detección, presioná **"Guardar detección actual"**.
2. El par (imagen, máscara detectada) se guarda en `data/feedback/`.
3. El contador de **"Pares acumulados"** aumenta.

> **Cuántos necesito:** Al menos 2 pares para poder reentrenar (se requiere split train/val). Se recomienda acumular 10 o más de imágenes variadas (distintas fechas, condiciones de luz) para resultados de calidad.

### Reentrenar el modelo U-Net

1. Con al menos 2 pares acumulados, el botón **"Reentrenar modelo U-Net"** se habilita.
2. Presionalo para iniciar el entrenamiento (puede tardar varios minutos sin GPU).
3. Al terminar, muestra la mejor val_loss obtenida.
4. El modelo queda guardado en `data/models/unet_best.pth`.
5. En la próxima detección, podés seleccionar **"U-Net (Deep Learning)"** como motor.

> **Tip:** Reentrenar periódicamente a medida que acumulás más feedback mejora progresivamente los resultados.

---

## 11. Limitaciones técnicas

| Limitación | Detalle |
|------------|---------|
| **Resolución 10 m/px** | Techos menores a ~50 m² son difíciles de detectar. Para Luján (techos residenciales de 60–300 m²) la resolución es adecuada. |
| **Nubosidad** | Imágenes con > 20% de nubes pueden reducir la calidad. Usar SCL para enmascarar. |
| **Materiales inusuales** | Techos verdes (vegetados) pueden ser confundidos como no-techos por el motor clásico. |
| **Sombras** | Edificios altos proyectan sombras que pueden generar falsos negativos. |
| **CRS** | El sistema trabaja en EPSG:32720 (UTM zona 20S). Imágenes en otros CRS se reproyectan automáticamente. |
| **Sin GPU** | U-Net en CPU puede tardar 1–3 minutos por análisis en imágenes grandes. |

---

## 12. Solución de problemas frecuentes

### "No se pudo cargar la imagen"

- Verificá que el archivo sea un GeoTIFF válido con al menos 6 bandas.
- Usá QGIS para verificar: `Raster → Información del raster`.
- Si el archivo tiene CRS no definido, asignalo en QGIS antes de cargar.

### "No se encontraron pesos entrenados para U-Net"

El modelo U-Net necesita ser entrenado antes de usarlo. Seguí estos pasos:
1. Ejecutá al menos 2 análisis con el **motor Clásico**.
2. Para cada uno, guardá la detección con **"Guardar detección actual"** en la pestaña Validación.
3. Presioná **"Reentrenar modelo U-Net"**.

### "No se pudo geocodificar la dirección"

- Verificá la conexión a internet.
- Usá un formato más específico: `"Av. Mitre 123, Luján, Buenos Aires, Argentina"`.
- Si el error persiste, la geocodificación usa Nominatim/OSM que puede tener datos incompletos para algunas zonas.

### "Error de localización: WMS"

El servidor WMS de CartoARBA puede estar temporalmente no disponible. El overlay de parcelas es opcional — la detección funciona sin él.

### La detección es muy lenta

- Considerá usar imágenes recortadas al área de interés (< 1 km²) en lugar de escenas completas.
- Imágenes más pequeñas (< 500×500 px) se procesan en segundos.
- Con GPU, la U-Net es 5–10× más rápida.

### Los resultados tienen muchos falsos positivos

Con el motor clásico:
- Aumentá el umbral **NDVI máx** para filtrar más vegetación.
- Disminuí el umbral **NDBI mín** para ser más exigente con superficies construidas.
- Aumentá el **Área mín (px)** para eliminar objetos pequeños.

Con U-Net:
- Acumulá más feedback con máscaras corregidas y reentrenar.

---

## Contacto y soporte

Este software es de uso interno. Para reportar problemas o sugerencias, contactar al desarrollador del proyecto.

*Documentación correspondiente a RoofScan v1.0 — 2026*
