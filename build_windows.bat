@echo off
REM ==========================================================================
REM build_windows.bat — Empaquetado de RoofScan para Windows con PyInstaller
REM ==========================================================================
REM
REM Uso:
REM   1. Activar entorno virtual: .venv\Scripts\activate  (o entorno conda)
REM   2. Ejecutar este script: build_windows.bat
REM   3. El ejecutable estará en: dist\RoofScan\RoofScan.exe
REM
REM Requisitos:
REM   pip install pyinstaller
REM   Todas las dependencias en requirements.txt instaladas
REM ==========================================================================

setlocal enabledelayedexpansion

echo.
echo ===================================================
echo  RoofScan - Build para Windows
echo ===================================================
echo.

REM Verificar que PyInstaller está instalado
pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PyInstaller no encontrado. Ejecuta:
    echo         pip install pyinstaller
    pause
    exit /b 1
)

REM Verificar que la app corre (smoke test)
echo [1/4] Verificando que la aplicacion puede importarse...
python -c "import roofscan; print('  OK: roofscan importado')" 2>&1
if errorlevel 1 (
    echo [ERROR] No se pudo importar roofscan. Verifica las dependencias.
    pause
    exit /b 1
)

REM Limpiar builds anteriores
echo [2/4] Limpiando builds anteriores...
if exist "dist\RoofScan" rmdir /s /q "dist\RoofScan"
if exist "build\RoofScan" rmdir /s /q "build\RoofScan"

REM Crear carpetas de datos vacías si no existen
echo [3/4] Preparando estructura de datos...
if not exist "roofscan\data\models" mkdir "roofscan\data\models"
if not exist "roofscan\data\feedback\images" mkdir "roofscan\data\feedback\images"
if not exist "roofscan\data\feedback\masks" mkdir "roofscan\data\feedback\masks"
if not exist "roofscan\data\cache" mkdir "roofscan\data\cache"

REM Compilar
echo [4/4] Compilando con PyInstaller...
pyinstaller roofscan.spec --clean --noconfirm

if errorlevel 1 (
    echo.
    echo [ERROR] La compilacion fallo. Revisa los mensajes anteriores.
    pause
    exit /b 1
)

echo.
echo ===================================================
echo  Build exitoso!
echo  Ejecutable: dist\RoofScan\RoofScan.exe
echo ===================================================
echo.
echo Para distribuir, comprime la carpeta dist\RoofScan\
echo y comparte el ZIP. El usuario solo necesita extraerlo
echo y ejecutar RoofScan.exe (no requiere instalar Python).
echo.

pause
