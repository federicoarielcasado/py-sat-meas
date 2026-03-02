# Setup del entorno de desarrollo — RoofScan

## Requisitos previos

- Windows 10/11 (64-bit)
- Python 3.10 o superior
- Cuenta gratuita en Copernicus Data Space Ecosystem (CDSE) — ver paso 2

---

## 1. Crear y activar el entorno virtual

```bash
# En la raíz del proyecto
python -m venv venv
venv\Scripts\activate
```

## 2. Registrarse en CDSE (solo una vez)

1. Ir a [https://dataspace.copernicus.eu](https://dataspace.copernicus.eu)
2. Hacer clic en **Sign Up** y completar el formulario (es gratuito)
3. Confirmar el email

> Las credenciales CDSE son las mismas para buscar y descargar imágenes Sentinel-2.
> No se requieren credenciales de pago ni aprobación especial.

## 3. Configurar las credenciales

```bash
# Copiar el ejemplo
copy .env.example .env
```

Abrir `.env` y completar:
```
CDSE_USER=tu_email@ejemplo.com
CDSE_PASSWORD=tu_contraseña
```

> **Importante:** El archivo `.env` está en `.gitignore` y nunca se sube al repositorio.

## 4. Instalar dependencias

```bash
# Dependencias de producción
pip install -r requirements.txt

# O en modo desarrollo (incluye pytest)
pip install -r requirements-dev.txt
```

> **Nota para Windows:** Si `rasterio` falla al instalar con pip, instalar desde wheels precompilados:
> ```bash
> pip install rasterio --find-links https://github.com/cgohlke/rasterio-wheels/releases
> ```
> O usar conda:
> ```bash
> conda install -c conda-forge rasterio geopandas pyproj
> ```

## 5. Instalar el paquete en modo desarrollo

```bash
pip install -e .
```

## 6. Verificar la instalación

```bash
python -c "from roofscan.core.ingesta.loader import load_geotiff; print('OK')"
```

## 7. Ejecutar los tests

Tests sin red (no requieren credenciales):
```bash
pytest tests/ -m "not network" -v
```

Tests con la API de CDSE (requieren `.env` configurado):
```bash
pytest tests/ -m network -v
```

---

## Resolución de problemas comunes

| Problema | Solución |
|----------|----------|
| `ModuleNotFoundError: rasterio` | Instalar via conda o wheels precompilados (ver paso 4) |
| `EnvironmentError: Credenciales CDSE no configuradas` | Verificar que `.env` existe y tiene `CDSE_USER` y `CDSE_PASSWORD` |
| `ConnectionError` al buscar escenas | Verificar conexión a internet; las credenciales pueden haber expirado |
| `FileNotFoundError` al cargar GeoTIFF | Verificar que la ruta es correcta y el archivo existe |
| `ValueError: no CRS` | El raster no tiene CRS. Asignarlo con QGIS: Layer → Properties → Source → CRS |
