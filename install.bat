@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title Porci-Integral - Instalador del Modulo de Inteligencia Artificial
cd /d "%~dp0"

echo =========================================================================
echo    Porci-Integral - Instalador del Modulo de Inteligencia Artificial
echo    (FastAPI + MobileNetV2 + YOLOv8 + Farrowing AI + PigSound + CCTV)
echo =========================================================================
echo.

REM 1. Verificar si Python esta disponible y funcional en el PATH
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR CRITICO] Python no esta instalado o no se encuentra en el PATH.
    echo.
    echo Por favor, instala Python 3.10 o superior ^(recomendado 3.10 - 3.12^):
    echo   1. Descarga Python desde: https://www.python.org/downloads/
    echo   2. Durante la instalacion, MARCA obligatoriamente la casilla:
    echo      "Add python.exe to PATH"
    echo   3. Cierra esta ventana y vuelve a ejecutar install.bat.
    echo.
    pause
    exit /b 1
)

REM Mostrar version detectada de Python y validar version minima (3.10+)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
for /f "tokens=1,2 delims=." %%a in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)

echo [OK] Entorno Python detectado: %PY_VER%

if %PY_MAJOR% lss 3 (
    echo [ERROR CRITICO] Se requiere Python 3.10 o superior. Version actual: %PY_VER%
    echo.
    pause
    exit /b 1
)
if %PY_MAJOR% equ 3 if %PY_MINOR% lss 10 (
    echo [ERROR CRITICO] Se requiere Python 3.10 o superior. Version actual: %PY_VER%
    echo.
    pause
    exit /b 1
)

REM Verificar existencia de requirements.txt
if not exist "requirements.txt" (
    echo [ERROR CRITICO] No se encontro "requirements.txt" en esta carpeta.
    echo Asegurate de ejecutar install.bat desde la raiz del modulo IA de Porci-Integral.
    echo.
    pause
    exit /b 1
)

REM 2. Asegurar existencia de directorios de trabajo del modulo IA
for %%D in (logs model_backups training_tasks output_v2) do (
    if not exist "%%D" (
        mkdir "%%D"
        echo [OK] Directorio creado: %%D\
    )
)

REM 3. Asegurar archivo de configuracion .env
if not exist ".env" (
    if exist "env.example" (
        copy /Y "env.example" ".env" >nul
        echo [OK] Archivo .env creado a partir de env.example.
    ) else (
        (
            echo # Ruta al dataset de fotos de animales ^(storage del backend Laravel^)
            echo DATASET_PATH=C:\laragon\www\Porci-Integral-backend\storage\app\public\fotos_animales
            echo # API Key para autenticacion con el backend ^(opcional^)
            echo AI_API_KEY=
        ) > ".env"
        echo [OK] Archivo .env generado con valores por defecto.
    )
) else (
    echo [OK] Archivo de configuracion .env detectado.
)

REM 4. Crear o validar entorno virtual aislado (.venv)
set "VENV_DIR=.venv"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"

echo.
echo [1/4] Configurando entorno virtual aislado (.venv)...
if exist "%VENV_PY%" (
    "%VENV_PY%" --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ADVERTENCIA] El entorno .venv parece danado o movido de equipo. Recreando...
        rmdir /s /q "%VENV_DIR%"
    )
)

if not exist "%VENV_PY%" (
    python -m venv "%VENV_DIR%"
    if !errorlevel! neq 0 (
        echo.
        echo [ERROR] No se pudo crear el entorno virtual .venv.
        echo Verifica permisos de escritura en esta carpeta y el modulo venv de Python.
        echo.
        pause
        exit /b 1
    )
    echo [OK] Entorno virtual creado en .venv\
) else (
    echo [OK] Entorno virtual .venv\ detectado y operativo.
)

echo.
echo [2/4] Actualizando gestor de paquetes pip en .venv...
"%VENV_PY%" -m pip install --upgrade pip

echo.
echo [3/4] Instalando dependencias de IA y API desde requirements.txt...
echo       (Este paso puede tardar varios minutos la primera vez)
"%VENV_PY%" -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Fallo al instalar las dependencias con pip.
    echo Posibles causas:
    echo   - Falta de conexion a internet en este equipo.
    echo   - Restriccion de proxy o firewall de red.
    echo   - Espacio insuficiente en disco ^(TensorFlow + YOLOv8 requieren ~2-3 GB^).
    echo   - Version de Python incompatible con TensorFlow/PyTorch ^(recomendado 3.10 - 3.12^).
    echo   - Permisos insuficientes en la carpeta del proyecto.
    echo.
    pause
    exit /b 1
)

echo.
echo [4/4] Verificando integridad de librerias de IA, vision, audio y API...
set "VERIFY_ERR=0"

"%VENV_PY%" -c "import fastapi; print('      [OK] fastapi: ' + fastapi.__version__)"
if %errorlevel% neq 0 set "VERIFY_ERR=1"

"%VENV_PY%" -c "import uvicorn; print('      [OK] uvicorn: ' + uvicorn.__version__)"
if %errorlevel% neq 0 set "VERIFY_ERR=1"

"%VENV_PY%" -c "import pydantic; print('      [OK] pydantic: ' + pydantic.__version__)"
if %errorlevel% neq 0 set "VERIFY_ERR=1"

"%VENV_PY%" -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; gpus=len(tf.config.list_physical_devices('GPU')); print(f'      [OK] tensorflow: {tf.__version__} (GPUs: {gpus})')"
if %errorlevel% neq 0 set "VERIFY_ERR=1"

"%VENV_PY%" -c "import torch; print(f'      [OK] torch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
if %errorlevel% neq 0 set "VERIFY_ERR=1"

"%VENV_PY%" -c "import ultralytics; print('      [OK] ultralytics (YOLOv8): ' + ultralytics.__version__)"
if %errorlevel% neq 0 set "VERIFY_ERR=1"

"%VENV_PY%" -c "import cv2; print('      [OK] opencv-python: ' + cv2.__version__)"
if %errorlevel% neq 0 set "VERIFY_ERR=1"

"%VENV_PY%" -c "import PIL; print('      [OK] Pillow: ' + PIL.__version__)"
if %errorlevel% neq 0 set "VERIFY_ERR=1"

"%VENV_PY%" -c "import numpy; print('      [OK] numpy: ' + numpy.__version__)"
if %errorlevel% neq 0 set "VERIFY_ERR=1"

"%VENV_PY%" -c "import sklearn; print('      [OK] scikit-learn: ' + sklearn.__version__)"
if %errorlevel% neq 0 set "VERIFY_ERR=1"

"%VENV_PY%" -c "import scipy; print('      [OK] scipy: ' + scipy.__version__)"
if %errorlevel% neq 0 set "VERIFY_ERR=1"

"%VENV_PY%" -c "import psutil; print('      [OK] psutil: ' + psutil.__version__)"
if %errorlevel% neq 0 set "VERIFY_ERR=1"

"%VENV_PY%" -c "import slowapi; print('      [OK] slowapi: instalado')"
if %errorlevel% neq 0 set "VERIFY_ERR=1"

echo.
echo Verificando modelos, clases y configuracion de Porci-Integral...

if exist "modelo_identificacion_cerdos.h5" (
    echo       [OK] Modelo MobileNetV2: modelo_identificacion_cerdos.h5
) else (
    echo       [ADVERTENCIA] Falta modelo_identificacion_cerdos.h5 ^(entrenar o copiar modelo^).
)

if exist "yolov8n.pt" (
    echo       [OK] Modelo YOLOv8: yolov8n.pt
) else (
    echo       [ADVERTENCIA] Falta yolov8n.pt ^(Ultralytics intentara descargarlo al iniciar^).
)

if exist "classes.json" (
    echo       [OK] Catalogo de clases: classes.json
) else (
    echo       [ADVERTENCIA] Falta classes.json ^(se generara al entrenar el modelo^).
)

"%VENV_PY%" -c "import os, pathlib; f=pathlib.Path('.env'); vals=[l.split('=',1)[1].strip().strip(chr(34)+chr(39)) for l in (f.read_text(encoding='utf-8').splitlines() if f.exists() else []) if l.strip().startswith('DATASET_PATH=')]; p=vals[0] if vals else ''; print('      [OK] DATASET_PATH: ' + p) if (p and os.path.exists(p)) else print('      [ADVERTENCIA] DATASET_PATH no encontrado en disco: ' + (p or '(vacio)'))"

echo.
if "%VERIFY_ERR%"=="0" (
    echo =========================================================================
    echo  INSTALACION DEL MODULO IA DE PORCI-INTEGRAL COMPLETADA CON EXITO
    echo =========================================================================
) else (
    echo =========================================================================
    echo  INSTALACION FINALIZADA CON ERRORES EN LIBRERIAS ^(revisa arriba^)
    echo =========================================================================
    echo.
    echo Corrige los errores de importacion antes de iniciar el servidor.
    echo.
    pause
    exit /b 1
)
echo.
echo Endpoints disponibles al iniciar el servicio:
echo   - API Base:   http://127.0.0.1:8000
echo   - Swagger UI: http://127.0.0.1:8000/docs
echo   - Telemetria: http://127.0.0.1:8000/salud
echo.
echo Iniciando servidor FastAPI de Porci-Integral en http://0.0.0.0:8000 ...
echo Presiona Ctrl+C para detener el servicio.
echo.
"%VENV_PY%" -m uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
echo.
echo El servidor FastAPI se ha detenido.
pause

endlocal
