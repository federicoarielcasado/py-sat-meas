@echo off
REM ==========================================================================
REM build_conda.bat — Alternativa con conda-pack (más simple que PyInstaller)
REM ==========================================================================
REM
REM conda-pack empaqueta el entorno conda completo como un ZIP/tar.gz.
REM Es más robusto con dependencias nativas (GDAL, rasterio, etc.) pero
REM produce un archivo más grande (~2-4 GB).
REM
REM Uso:
REM   1. Asegurarse de tener el entorno conda activo: conda activate roofscan
REM   2. Ejecutar: build_conda.bat
REM   3. Compartir: roofscan_env.zip + la carpeta del proyecto
REM
REM En la PC destino:
REM   1. Extraer roofscan_env.zip en C:\roofscan_env\
REM   2. Ejecutar: C:\roofscan_env\python.exe -m roofscan
REM ==========================================================================

setlocal

echo.
echo ===================================================
echo  RoofScan - Build con conda-pack
echo ===================================================
echo.

REM Verificar conda-pack
conda run -n base conda-pack --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Instalando conda-pack...
    conda install -n base -c conda-forge conda-pack -y
)

REM Nombre del entorno actual
set ENV_NAME=roofscan
echo [1/2] Empaquetando entorno conda '%ENV_NAME%'...
echo       (puede tardar varios minutos)

conda pack -n %ENV_NAME% -o roofscan_env.zip --ignore-editable-packages

if errorlevel 1 (
    echo [ERROR] conda-pack fallo. Verificar que el entorno '%ENV_NAME%' existe:
    echo         conda env list
    pause
    exit /b 1
)

echo.
echo [2/2] Creando launcher...

REM Crear un .bat de lanzamiento para la PC destino
(
echo @echo off
echo set CONDA_ENV=%~dp0roofscan_env
echo "%CONDA_ENV%\python.exe" -m roofscan
) > launch_roofscan.bat

echo.
echo ===================================================
echo  Listo!
echo  Archivos a distribuir:
echo    - roofscan_env.zip  (entorno Python completo)
echo    - launch_roofscan.bat  (lanzador)
echo    - Carpeta del proyecto (codigo fuente)
echo ===================================================
echo.
echo En la PC destino:
echo   1. Extraer roofscan_env.zip en la misma carpeta
echo   2. Ejecutar launch_roofscan.bat
echo.

pause
