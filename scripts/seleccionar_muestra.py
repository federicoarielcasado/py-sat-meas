"""Selecciona 20 parcelas representativas de Luján para validación manual.

Estratifica por rango de área para cubrir el espectro completo de tipos de
techo (pequeño residencial → gran industrial). Genera URLs de Google Maps
satellite para facilitar la medición visual y un CSV pre-formateado para
completar área manual y tipo de estructura.

Uso:
    # Muestra estándar de 20 parcelas urbanas
    python scripts/seleccionar_muestra.py

    # Personalizar cantidad o fuente
    python scripts/seleccionar_muestra.py --gpkg data/catastro/lujan_parcelas.gpkg --n 20

Salida:
    data/validacion/muestra_lujan.csv

Columnas de salida:
    cca              Código catastral único
    partida          Número de partida ARBA
    tipo_catastral   Urbano / Rural (del catastro)
    estrato          Rango de área asignado
    area_catastral   Área declarada en catastro (m²)
    lat / lon        Centroide WGS84
    google_maps_url  URL para abrir en Google Maps satellite (zoom 19)
    area_manual_m2   [COMPLETAR] área medida manualmente
    tipo_manual      [COMPLETAR] VIVIENDA / GALPON / INDUSTRIAL / OTRO
    notas            [COMPLETAR] observaciones libres
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("seleccionar_muestra")

_GPKG_DEFAULT = Path("data/catastro/lujan_parcelas.gpkg")
_OUTPUT_DIR = Path("data/validacion")
_OUTPUT_CSV = "muestra_lujan.csv"
_N_DEFAULT = 20
_SEED = 42

# Estratos: (nombre, área_min_m2, área_max_m2, n_parcelas)
# Diseñados para cubrir el rango residencial de Luján:
#   - pequeña:    4-6 px en Sentinel-2; casas chicas, garajes
#   - mediana:    6-16 px; vivienda residencial típica
#   - grande:     >16 px; casas grandes, comercios
#   - industrial: galpones, naves, depósitos
_ESTRATOS = [
    ("pequeña",     0,      150,   4),
    ("mediana",   150,      400,   7),
    ("grande",    400,    1_500,   6),
    ("industrial", 1_500, float("inf"), 3),
]

# Rubrica de clasificación manual (para referencia en el CSV)
_RUBRICA = (
    "VIVIENDA: área ≤ 400 m², forma compacta, uso residencial. "
    "GALPON: área > 300 m², muy elongado (largo/ancho > 2). "
    "INDUSTRIAL: área > 1500 m², naves o depósitos. "
    "OTRO: garajes, estructuras irregulares, mixto."
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--gpkg", type=Path, default=_GPKG_DEFAULT,
        help=f"GeoPackage catastral (default: {_GPKG_DEFAULT}).",
    )
    p.add_argument(
        "--n", type=int, default=_N_DEFAULT,
        help=f"Número de parcelas a seleccionar (default: {_N_DEFAULT}).",
    )
    p.add_argument(
        "--seed", type=int, default=_SEED,
        help="Semilla para reproducibilidad del muestreo.",
    )
    p.add_argument(
        "--incluir-rural", action="store_true",
        help="Incluir también parcelas rurales (default: solo urbanas).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        import geopandas as gpd
        import pandas as pd
    except ImportError:
        log.error("Instalar dependencias: pip install geopandas pandas")
        sys.exit(1)

    if not args.gpkg.exists():
        log.error("No se encontró el catastro: %s", args.gpkg)
        log.error("Ejecutar primero: python scripts/download_catastro.py")
        sys.exit(1)

    # --- Cargar catastro ---
    log.info("Cargando catastro: %s", args.gpkg)
    gdf = gpd.read_file(str(args.gpkg))
    log.info("  %d parcelas totales. Columnas: %s", len(gdf), list(gdf.columns))

    # --- Filtrar por tipo ---
    if "tpa" in gdf.columns and not args.incluir_rural:
        mask = gdf["tpa"].str.upper().str.startswith("U")
        gdf_f = gdf[mask].copy()
        log.info("  %d parcelas urbanas (filtro tpa startswith 'U')", len(gdf_f))
    else:
        gdf_f = gdf.copy()
        if args.incluir_rural:
            log.info("  Usando todas las parcelas (--incluir-rural activo)")

    if len(gdf_f) == 0:
        log.error("No quedan parcelas después del filtro.")
        sys.exit(1)

    # --- Calcular área en m² ---
    if "ara1" in gdf_f.columns:
        gdf_f["_area_m2"] = gdf_f["ara1"].astype(float)
        log.info("  Área tomada de columna 'ara1' (declarada en catastro).")
    else:
        log.info("  Columna 'ara1' no encontrada — calculando desde geometría.")
        gdf_metric = gdf_f.to_crs("EPSG:32720")
        gdf_f["_area_m2"] = gdf_metric.geometry.area

    # Descartar parcelas con área cero (datos inválidos)
    gdf_f = gdf_f[gdf_f["_area_m2"] > 0].copy()

    # --- Calcular centroides WGS84 ---
    centroids = gdf_f.geometry.centroid
    gdf_f["_lat"] = centroids.y.round(7)
    gdf_f["_lon"] = centroids.x.round(7)

    # --- Estratificar y muestrear ---
    muestras = []
    log.info("")
    log.info("Estratificación por área:")
    for nombre, a_min, a_max, n_est in _ESTRATOS:
        subset = gdf_f[
            (gdf_f["_area_m2"] >= a_min) & (gdf_f["_area_m2"] < a_max)
        ]
        n_real = min(n_est, len(subset))
        a_max_str = f"{int(a_max)}" if a_max != float("inf") else "∞"
        if n_real == 0:
            log.warning(
                "  %-12s [%5d – %5s m²]: sin parcelas disponibles",
                nombre, a_min, a_max_str,
            )
            continue
        sample = subset.sample(n=n_real, random_state=args.seed).copy()
        sample["_estrato"] = nombre
        muestras.append(sample)
        log.info(
            "  %-12s [%5d – %5s m²]: %d parcelas seleccionadas de %d disponibles",
            nombre, a_min, a_max_str, n_real, len(subset),
        )

    if not muestras:
        log.error("No se pudo seleccionar ninguna parcela.")
        sys.exit(1)

    combined = pd.concat(muestras, ignore_index=True)
    log.info("  Total: %d parcelas", len(combined))

    # --- Construir CSV de salida ---
    out = pd.DataFrame()

    # Identificadores
    if "cca" in combined.columns:
        out["cca"] = combined["cca"].values
    if "pda" in combined.columns:
        out["partida"] = combined["pda"].values
    if "tpa" in combined.columns:
        out["tipo_catastral"] = combined["tpa"].values

    out["estrato"] = combined["_estrato"].values
    out["area_catastral_m2"] = combined["_area_m2"].round(1).values
    out["lat"] = combined["_lat"].values
    out["lon"] = combined["_lon"].values

    # URL Google Maps satelital zoom 19
    out["google_maps_url"] = [
        f"https://maps.google.com/?q={lat},{lon}&t=k&z=19"
        for lat, lon in zip(out["lat"], out["lon"])
    ]

    # Columnas vacías para llenado manual
    out["area_manual_m2"] = ""
    out["tipo_manual"] = ""   # VIVIENDA / GALPON / INDUSTRIAL / OTRO
    out["notas"] = ""

    # --- Guardar ---
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUTPUT_DIR / _OUTPUT_CSV
    out.to_csv(str(out_path), index=False)

    log.info("")
    log.info("=" * 65)
    log.info("Muestra guardada: %s", out_path)
    log.info("")
    log.info("Instrucciones de medición manual:")
    log.info("  1. Abrí el CSV en Excel o Google Sheets.")
    log.info("  2. Hacé clic en la URL de cada parcela → Google Maps satellite.")
    log.info("  3. Clic derecho en el mapa → 'Medir distancias'.")
    log.info("     Trazá el contorno del techo para obtener el área en m².")
    log.info("  4. Completá 'area_manual_m2' con el valor medido.")
    log.info("  5. Completá 'tipo_manual' según la rubrica:")
    log.info("       VIVIENDA  → casa residencial (compacta, ≤ 400 m²)")
    log.info("       GALPON    → elongado, chapa, > 300 m²")
    log.info("       INDUSTRIAL→ nave/depósito grande, > 1500 m²")
    log.info("       OTRO      → garaje, estructura irregular, mixto")
    log.info("  6. Guardá el CSV completado.")
    log.info("")
    log.info("Para ejecutar la mensura solo sobre estas parcelas:")
    log.info("")
    log.info("  python scripts/batch_mensura.py \\")
    log.info("      --image data/cache/<S2A_..._stacked.tif> \\")
    log.info("      --parcelas %s \\", args.gpkg)
    log.info("      --partidas %s \\", out_path)
    log.info("      --output data/output/mensura_muestra.csv \\")
    log.info("      --output-geojson --classify")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
