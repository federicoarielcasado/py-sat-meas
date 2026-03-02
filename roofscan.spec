# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec para RoofScan
# Uso: pyinstaller roofscan.spec --clean --noconfirm
#
# Requisitos antes de compilar:
#   1. Activar el entorno conda/venv con todas las deps instaladas
#   2. Verificar que "python -m roofscan" corre correctamente
#   3. pyinstaller debe estar instalado: pip install pyinstaller

from pathlib import Path
import sys

block_cipher = None

# Archivos de datos a incluir (src_path, dest_folder_in_bundle)
added_datas = [
    # Incluir modelos entrenados si existen
    ("roofscan/data/models", "roofscan/data/models"),
    # Carpeta de feedback vacía para que exista en la distribución
    ("roofscan/data/feedback", "roofscan/data/feedback"),
    # Plantilla de .env para el usuario
    (".env.example", "."),
    # Documentación
    ("docs/user_guide.md", "docs"),
]

# Importaciones ocultas que PyInstaller no detecta automáticamente
hidden_imports = [
    # rasterio
    "rasterio._shim",
    "rasterio.control",
    "rasterio.crs",
    "rasterio._env",
    "rasterio._warp",
    "rasterio.features",
    "rasterio.transform",
    # fiona (necesario para geopandas)
    "fiona",
    "fiona.ogrext",
    "fiona._shim",
    # pyproj
    "pyproj",
    "pyproj.transformer",
    "pyproj._datadir",
    # shapely
    "shapely",
    "shapely.geometry",
    # geopandas
    "geopandas",
    "geopandas.io.file",
    # torch / smp (solo si U-Net disponible)
    "torch",
    "torch.nn",
    "segmentation_models_pytorch",
    "timm",
    "timm.models",
    # scipy
    "scipy.ndimage",
    "scipy.ndimage._filters",
    # skimage
    "skimage.morphology",
    # PyQt6
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    # matplotlib backend para Qt
    "matplotlib.backends.backend_qt5agg",
    "matplotlib.backends.backend_agg",
    # requests
    "requests",
    "requests.adapters",
    # PIL
    "PIL",
    "PIL.Image",
    # cdsetool
    "cdsetool",
    # dotenv
    "dotenv",
]

a = Analysis(
    ["roofscan/__main__.py"],
    pathex=["."],
    binaries=[
        # Si GDAL no se detecta automáticamente, agregar las DLLs aquí:
        # (r"C:\conda\envs\roofscan\Library\bin\gdal*.dll", "."),
        # (r"C:\conda\envs\roofscan\Library\bin\proj*.dll", "."),
    ],
    datas=added_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Excluir módulos pesados innecesarios
        "matplotlib.tests",
        "numpy.tests",
        "scipy.tests",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "tk",
        "tkinter",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RoofScan",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,           # False = app de escritorio sin ventana de consola
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon="docs/icon.ico",  # Descomentar si se provee un ícono .ico
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="RoofScan",
)
