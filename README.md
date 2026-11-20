# Porci-Integral: Servicio de IA para Reconocimiento de Animales

Servicio de inteligencia artificial para identificación biométrica de cerdas usando **MobileNetV2** + **YOLOv8**.

## Stack Tecnológico

| Componente         | Tecnología                        |
| ------------------ | --------------------------------- |
| API                | FastAPI + Uvicorn                 |
| Clasificación      | TensorFlow / Keras (MobileNetV2)  |
| Detección          | YOLOv8 (Ultralytics)              |
| Monitoreo CCTV     | OpenCV + RTSP + YOLO + MobileNet  |

## Estructura

```
Reconocimiento-Cerdos-IA/
├── app_fastapi.py                 # API principal (FastAPI)
├── entrenar_v2.py                 # Entrenamiento MobileNetV2 (2 fases)
├── training_manager.py            # Gestor de entrenamiento async
├── cctv_monitor.py                # Monitoreo CCTV en tiempo real
├── modelo_identificacion_cerdos.h5     # Modelo en producción
├── model_backups/                 # Backups automáticos del modelo (max 3)
├── output_v2/                     # Modelos entrenados + classes.json
├── training_tasks/                # Progreso de entrenamientos (auto-limpieza)
├── .env                           # Configuración (DATASET_PATH)
└── env.example                    # Plantilla para .env
```

## Iniciar el Servicio

```bash
.venv\Scripts\activate
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
```

## Endpoints

### Generales & MLOps
| Endpoint                  | Método | Descripción                          |
| ------------------------- | ------ | ------------------------------------ |
| `/`                       | GET    | Health check simple                  |
| `/salud`                  | GET    | Estado completo, telemetría hardware (CPU, RAM) y modelos |
| `/clases`                 | GET    | Lista de clases disponibles          |
| `/reconocer`              | POST   | Reconocer cerda por imagen (file) con latencia (ms) |
| `/reconocer_base64`       | POST   | Reconocer cerda por imagen (base64) con latencia (ms) |
| `/reconocer-completo`     | POST   | Pipeline 2 fases (YOLOv8 + MobileNetV2) con desglose de latencias |
| `/detectar`               | POST   | Detectar cerdas en imagen (YOLO)     |
| `/detector/status`        | GET    | Estado del detector YOLO             |
| `/exportar-foto`          | POST   | Guardar foto en el dataset           |
| `/confirmar`              | POST   | Registrar confirmación de IA         |
| `/detectar-enfermedad`    | POST   | Evaluación de riesgo sanitario       |

### Dataset & Diagnóstico
| Endpoint                  | Método | Descripción                          |
| ------------------------- | ------ | ------------------------------------ |
| `/dataset-estadisticas`   | GET    | Estadísticas básicas del dataset     |
| `/dataset/diagnostico`    | GET    | Diagnóstico de desbalance zootécnico y alertas de muestra insuficiente (<30 fotos) |

### Acústica & Biometría Zootécnica (Octubre 2026)
| Endpoint                  | Método | Descripción                          |
| ------------------------- | ------ | ------------------------------------ |
| `/analizar_audio_tos`     | POST   | Detección acústica de tos respiratoria porcina mediante FFT (800–2400 Hz) |
| `/estimar_biometria_base64` | POST | Estimación óptica de biometría corporal y peso vivo sin balanza |

### Farrowing AI Monitor (Monitoreo de Maternidad y Partos)
| Endpoint                  | Método | Descripción                          |
| ------------------------- | ------ | ------------------------------------ |
| `/farrowing/camaras-config` | GET   | Lista de configuración de cámaras estándar de maternidad |
| `/farrowing/analizar`     | POST   | Inferencia de postura, lechones, distocia y aplastamiento por imagen (multipart) |
| `/farrowing/analizar-base64` | POST | Inferencia de maternidad por imagen en Base64 con telemetría en ms |

### Entrenamiento
| Endpoint                  | Método | Descripción                          |
| ------------------------- | ------ | ------------------------------------ |
| `/entrenar`               | POST   | Iniciar entrenamiento async          |
| `/entrenar/historial`     | GET    | Historial de entrenamientos          |
| `/entrenar/{task_id}`     | GET    | Estado del entrenamiento             |
| `/entrenar/activo`        | GET    | Entrenamiento activo (si existe)     |

### Modelo & Telemetría
| Endpoint                  | Método | Descripción                          |
| ------------------------- | ------ | ------------------------------------ |
| `/modelos/info`           | GET    | Arquitectura, tamaño, clases y registros históricos |
| `/modelo/recargar`        | POST   | Recargar modelo desde disco          |
| `/modelo/versiones`       | GET    | Lista backups disponibles            |
| `/modelo/restaurar`       | POST   | Restaurar un backup                  |

## Monitoreo Acústico de Tos Respiratoria (PigSound — FFT)

Algoritmo de procesamiento digital de señales de audio de galpón mediante Transformada Rápida de Fourier (FFT):

- **Banda de Frecuencia Porcina**: 800 Hz a 2400 Hz (frecuencia de resonancia traqueobronquial porcina).
- **Filtrado de Ruido Ambiental**: Supresión de ruidos de baja frecuencia ($<400$ Hz) generados por extractores, comederos mecánicos o pisadas.
- **Detección Precoz**: Identifica brotes de PRRS, *Mycoplasma hyopneumoniae* o *Actinobacillus pleuropneumoniae* 4 a 5 días antes de los síntomas clínicos macroscópicos.

```bash
# Análisis de muestra de audio por Base64
curl -X POST http://127.0.0.1:8000/analizar_audio_tos \
  -H "Content-Type: application/json" \
  -d '{"audio_base64": "<base64_pcm_wav>", "sample_rate": 16000}'
```

Respuesta típica:
```json
{
  "success": true,
  "probabilidad_tos": 0.88,
  "es_tos_patologica": true,
  "nivel_alerta": "alerta_tos",
  "frecuencia_pico_hz": 1450.0,
  "ratio_energia_tos": 0.462,
  "duracion_segundos": 1.5,
  "latencia_ms": 14.2
}
```

## Estimación Óptica de Biometría & Peso Vivo (PigVision)

Estimación de peso vivo y volumen corporal basada en regresión morfológica por visión artificial:

- **Eliminación del Estrés de Arreo**: Elimina mermas de peso asociadas al traslado forzado a básculas convencionales.
- **Relación Largo × Perímetro Torácico**: Relación alométrica entre superficie dorsal detectada y masa muscular del cerdo.

```bash
# Estimación de peso óptico por Base64
curl -X POST http://127.0.0.1:8000/estimar_biometria_base64 \
  -H "Content-Type: application/json" \
  -d '{"imagen_base64": "<base64_jpg_png>"}'
```

## Monitoreo de Pariciones por Visión Artificial (Farrowing AI Monitor)

Módulo zootécnico especializado para la detección automatizada de partos y riesgos en paritorios mediante visión por computadora:

- **Detección de Posturas de la Cerda**: Clasificación de `decubito_lateral` (postura óptima de parición), `decubito_esternal` y `de_pie`.
- **Conteo de Lechones**: Detección visual y recuento de lechones expulsados en la jaula paritoria.
- **Detección de Parto Activo**: Disparo de evento cuando la cerda se encuentra en decúbito lateral con lechones nacidos o contracciones visibles (probabilidad $\ge 80\%$).
- **Alerta Crítica de Distocia**: Detección de cese o retraso en la expulsión ($>55$ minutos sin avance tras inicio de trabajo de parto).
- **Alerta de Riesgo de Aplastamiento**: Detección de cambios bruscos de postura de la madre con lechones en proximidad inmediata.

```bash
# Análisis de frame por Base64
curl -X POST http://127.0.0.1:8000/farrowing/analizar-base64 \
  -H "Content-Type: application/json" \
  -d '{
    "camara_id": "cam-01",
    "image_base64": "<base64_data>",
    "tiempo_en_postura_min": 60,
    "dias_gestacion": 114,
    "lechones_previos": 2
  }'
```

## Detección Temprana de Enfermedades

```bash
curl -X POST http://127.0.0.1:8000/detectar-enfermedad \
  -H "Content-Type: application/json" \
  -d '{"animal_id": 5, "peso_actual_kg": 62, "peso_anterior_kg": 68, "dias_entre_pesajes": 10}'
```

Reglas de evaluación:
- Pérdida >5% en 7 días → ALTO
- Pérdida >3% en 7 días → MEDIO
- Sin ganancia en <90 días → MEDIO
- Temperatura ≥40°C → ALERTA

## Monitoreo CCTV en Tiempo Real

```bash
# Webcam local
python cctv_monitor.py --source 0

# Cámara IP (RTSP)
python cctv_monitor.py --source "rtsp://usuario:pass@camara:554/stream"

# Con confianza personalizada
python cctv_monitor.py --source 0 --confidence 0.6
```

Controles:
- `q` — Salir
- `p` — Pausar/Reanudar
- `s` — Guardar frame actual

Detecciones guardadas en `cctv_log.jsonl` para análisis posterior.

## Entrenamiento

```bash
curl -X POST http://127.0.0.1:8000/entrenar \
  -H "Content-Type: application/json" \
  -d '{"include_classes": ["cerda_001", "reproductor"]}'
```

### Fases
1. **Frozen (15 epochs)**: Base del modelo congelada
2. **Fine-tuning (20 epochs)**: Últimas 30 capas descongeladas

### Backup automático
Antes de cada entrenamiento se crea un backup. Máximo 3 versiones.

### Gestión de tareas
- Persisten en disco (`training_tasks/tasks_index.json`)
- Tareas "running" al reiniciar pasan a "error"
- `get_status()` recupera progreso desde disco si la tarea fue interrumpida por reinicio pero el subproceso sigue activo
- Limpieza automática (máximo 15 tareas completadas)

## Dataset

```
{DATASET_PATH}/
├── cerda_001/
├── cerda_002/
└── ...
```

Configurable via `DATASET_PATH` en `.env`.

## Notas

- **Tiempo estimado entrenamiento**: 30-60 min en CPU
- MobileNetV2 con pesos ImageNet
- Data augmentation: rotación, zoom, flip, brillo adaptado a galpones porcinos
- Class weighting automático con clipping seguro `[0.2, 10.0]` para evitar gradientes destructivos
- Test-Time Augmentation (TTA) vectorizado en batch único `(4, 224, 224, 3)` para inferencia ultra-rápida en CPU
- Telemetría en tiempo real: medición de latencia en milisegundos y monitoreo de RAM/CPU

## Integración con Módulos Especializados Porci-Integral (Noviembre 2026)

El microservicio de IA se integra de forma nativa con los nuevos subsistemas zootécnicos del ecosistema Porci-Integral:
- **HeatVision (Termografía Vulvar & Ventana de Inseminación)**: Compatible con feeds termográficos FLIR/Lepton para la detección de picos térmicos preovulatorios (ΔT ≥ 1.2 °C) y sincronización con dosis seminales de SpermLab.
- **Smart Piglet ThermalPad (Visión & Microclima)**: Detección óptica de índice de agrupamiento (*huddling* ≥ 65%) y alertas tempranas de aplastamiento bajo lámparas y placas térmicas en maternidad.
- **EcoPorc & PigSound (Acústica + Telemetría de Gases)**: Correlación cruzada entre índice de tos acústico (800–2400 Hz) y concentraciones de $NH_3$ (≥ 20 ppm) / $H_2S$ (≥ 10 ppm) en fosas profundas.

## Pruebas Automatizadas

El microservicio cuenta con una suite integral de 14 pruebas automatizadas con `pytest` que valida telemetría, inferencia, balanceo, monitoreo de partos, acústica y biometría:

```powershell
python -m pytest -v
```

Casos de prueba cubiertos (14 tests passed, 100%):
1. `test_farrowing_camaras_default`: Configuración estandarizada de cámaras en maternidad.
2. `test_farrowing_analyzer_parto_inminente`: Detección de signos preparto y decúbito.
3. `test_farrowing_analyzer_parto_activo_con_lechones`: Detección de parto en curso con conteo de lechones.
4. `test_farrowing_analyzer_alerta_distocia`: Detección y disparo de alerta de distocia (>55 min sin avance).
5. `test_endpoint_farrowing_camaras_config`: Endpoint GET de configuración de cámaras.
6. `test_endpoint_farrowing_analizar_base64`: Inferencia de parto en Base64 con telemetría en ms.
7. `test_salud_telemetria_hardware`: Telemetría de CPU y memoria RAM.
8. `test_clases_disponibles`: Catálogo de clases zootécnicas y umbral de confianza.
9. `test_dataset_diagnostico_zootecnico`: Identificación de clases críticas (<30 fotos) y cálculo de ratio de desbalance.
10. `test_modelos_info_tecnica`: Verificación de arquitectura y estado de YOLOv8.
11. `test_inferencia_reconocer_base64_con_latencia`: Predicción Top-3 con TTA y latencia en ms.
12. `test_seguridad_api_key_reconocer_completo`: Protección de endpoints sensibles con API Key.
13. `test_pipeline_completo_dos_fases_yolo_mobilenet`: Integración YOLOv8 + MobileNetV2 con reporte de latencias parciales.
14. `test_estimacion_biometria_peso_y_bcs`: Estimación de biometría corporal, peso alométrico y condición corporal.

## Setup Portátil (sin Laragon)

### Requisitos
- Python 3.10+ (descargar de python.org)
- Pip instalado

### Instalación

```powershell
# 1. Crear entorno virtual
python -m venv .venv

# 2. Activar
.\.venv\Scripts\Activate.ps1   # PowerShell
# o
.\.venv\Scripts\activate       # CMD

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno (crear .env)
cp env.example .env
```

### Configuración del .env

```env
# Ruta al dataset de fotos de animales (backend Laravel)
DATASET_PATH=C:\laragon\www\Porci-Integral-backend\storage\app\public\fotos_animales

# API Key para autenticación con backend (opcional)
AI_API_KEY=tu_api_key_aqui
```

### Iniciar Servicio

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
```

Verificar en: `http://127.0.0.1:8000/docs` (Swagger UI interactivo)

### Solución de Problemas

| Problema | Solución |
|----------|----------|
| `Fatal error in launcher` | Recrear venv: `rmdir /s /q .venv && python -m venv .venv` |
| `ExecutionPolicy` | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Modelo no carga | Verificar que `modelo_identificacion_cerdos.h5` existe en la raíz |
| Dataset no encontrado | Configurar `DATASET_PATH` en `.env` con ruta válida |
