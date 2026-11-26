"""
API FastAPI para Reconocimiento de Cerdas
=========================================
Servicio principal del módulo de IA. Expone endpoints para:
- Reconocimiento biométrico (MobileNetV2)
- Detección de objetos (YOLOv8)
- Gestión del dataset
- Entrenamiento asíncrono
- Backup y restauración de modelos

Uso: uvicorn app_fastapi:app --reload
"""

import os
import io
import json
import base64
import shutil
import threading
import re
import logging
import gc
import torch
from datetime import datetime
from time import time, perf_counter
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf
from pathlib import Path
from training_manager import manager, TASKS_DIR
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from ultralytics import YOLO
import utils
from farrowing_service import farrowing_analyzer, CAMARAS_DEFAULT


# ── Configuración ────────────────────────────────────────────────
utils.load_env()
NEGATIVE_CLASS = utils.NEGATIVE_CLASS

BASE_DIR = utils.BASE_DIR
MODEL_PATH = BASE_DIR / "modelo_identificacion_cerdos.h5"
MODELS_BACKUP_DIR = BASE_DIR / "model_backups"
MAX_BACKUPS = 3
IMG_SIZE = utils.IMG_SIZE
THRESHOLD = utils.load_threshold()

DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    r"C:\laragon\www\Porci-Integral-backend\storage\app\public\datasets\animales"
)
RECONOCIMIENTOS_PATH = os.environ.get(
    "RECONOCIMIENTOS_PATH",
    r"C:\laragon\www\Porci-Integral-backend\storage\app\public\datasets\reconocimientos"
)

MODELS_BACKUP_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Cache de estadísticas ────────────────────────────────────────
_stats_cache = {"data": None, "timestamp": 0}
CACHE_TTL = 60


# ── Funciones de Backup del Modelo ─────────────────────────────

def backup_current_model():
    """
    Crea una copia del modelo actual antes de entrenar.
    Mantiene hasta MAX_BACKUPS versiones (3 por defecto).
    También respalda classes.json asociado.
    """
    if not MODEL_PATH.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"modelo_{timestamp}.h5"
    backup_path = MODELS_BACKUP_DIR / backup_name
    shutil.copy2(str(MODEL_PATH), str(backup_path))

    # Elimina backups excedentes (los más viejos)
    backups = sorted(MODELS_BACKUP_DIR.glob("modelo_*.h5"), reverse=True)
    for old in backups[MAX_BACKUPS:]:
        old.unlink()

    classes_src = BASE_DIR / "classes.json"
    if classes_src.exists():
        backup_classes = MODELS_BACKUP_DIR / f"classes_{timestamp}.json"
        shutil.copy2(str(classes_src), str(backup_classes))

    return backup_name


def list_model_backups():
    """Retorna metadatos de todos los backups disponibles."""
    backups = []
    for f in sorted(MODELS_BACKUP_DIR.glob("modelo_*.h5"), reverse=True):
        ts = f.stem.replace("modelo_", "")
        stats = f.stat()
        size_mb = stats.st_size / (1024 * 1024)
        classes_file = MODELS_BACKUP_DIR / f"classes_{ts}.json"
        backups.append({
            "version": ts,
            "filename": f.name,
            "size_mb": round(size_mb, 1),
            "created": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "has_classes": classes_file.exists(),
        })
    return backups


# ── App FastAPI ─────────────────────────────────────────────────

app = FastAPI(
    title="API Reconocimiento de Cerdas",
    description="Sistema de identificación biométrica de cerdas",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://porci-integral-backend.test", "http://127.0.0.1:5173", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate Limiting ───────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

# ── API Key Auth ────────────────────────────────────────────────
API_KEY = os.environ.get("AI_API_KEY") or os.environ.get("API_KEY", "")

async def verify_admin_key(x_api_key: str = Header(None)):
    """
    Valida la autenticación de peticiones entrantes contra la clave compartida del ecosistema.

    @param x_api_key Cabecera HTTP X-API-Key enviada por Laravel o clientes autorizados.

    @returns {bool} True si la llave es válida.

    @author Jhon Javer Lozano Arce

    @date 19 de septiembre de 2026
    """
    if not API_KEY:
        return True  # Modo desarrollo sin llave configurada
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Acceso no autorizado: API Key inválida o no proporcionada")
    return True


# ── Carga del Modelo y Utilidades ──────────────────────────────

CLASS_NAMES = utils.load_classes()
THRESHOLD = utils.load_threshold()
logger.info(f"Clases cargadas: {len(CLASS_NAMES)} - {CLASS_NAMES}")

model_lock = threading.Lock()
model = None

# AUD-ML-004: Control de memory leaks en inferencia sobre CPU.
# Limpieza de grafos temporales y recolección forzada periódica para evitar saturación de RAM.
_inference_counter = 0
_INFERENCE_GC_INTERVAL = 25


def _limpiar_memoria_inferencia():
    """
    AUD-ML-004: Libera tensores temporales y fuerza recolección de basura cada N inferencias en CPU.
    """
    global _inference_counter
    _inference_counter += 1
    if _inference_counter % _INFERENCE_GC_INTERVAL == 0:
        gc.collect()


def load_model():
    """
    Carga el modelo MobileNetV2 desde disco mediante carga perezosa con double-checked locking.

    @returns {None}

    @author Jhon Javer Lozano Arce

    @date 19 de septiembre de 2026
    """
    global model
    if model is None:
        with model_lock:
            if model is None:  # Double-checked locking
                if os.path.exists(MODEL_PATH):
                    logger.info(f"Cargando modelo desde: {MODEL_PATH}")
                    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                    logger.info("Modelo cargado exitosamente")
                else:
                    logger.warning(f"Modelo no encontrado en: {MODEL_PATH}")


def preprocess_image(image_bytes):
    """
    Preprocesa una imagen en bytes a formato tensorial normalizado de MobileNetV2.

    @param image_bytes Bytes crudos del archivo de imagen.

    @returns {numpy.ndarray} Tensor normalizado con forma (1, 224, 224, 3).

    @author Jhon Javer Lozano Arce

    @date 19 de septiembre de 2026
    """
    return utils.preprocess_image(image_bytes)


def predict_top3(image_bytes, use_tta=True, threshold=None, filter_negative=False):
    """
    Ejecuta inferencia biométrica MobileNetV2 y retorna el Top-3 de predicciones zootécnicas con latencia.

    Aplica Test-Time Augmentation vectorizado en un solo batch sobre CPU para minimizar latencia
    y clasifica como 'Desconocido' cualquier inferencia inferior al umbral de confianza biológica.

    @param image_bytes Contenido binario de la imagen a clasificar.
    @param use_tta Bandera para activar aumentos en tiempo de test (TTA).
    @param threshold Umbral de confianza biológica personalizado (defecto THRESHOLD).

    @returns {list} Lista con el ranking Top-3 de identificaciones biométricas con latencia en ms.

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    load_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo MobileNetV2 no cargado en memoria")

    t0 = perf_counter()
    umbral_efectivo = threshold if (threshold is not None and threshold > 0) else THRESHOLD

    if use_tta:
        tta_inputs = utils.preprocess_image_tta(image_bytes)
        batch_array = np.vstack([aug_array for aug_array, _label in tta_inputs])
        with model_lock:
            # AUD-ML-004: Invocación directa funcional model(tensor, training=False)
            # Evita crear trazas tf.Graph por batch y elimina fugas de memoria de model.predict en bucles CPU.
            batch_tensor = tf.convert_to_tensor(batch_array, dtype=tf.float32)
            batch_preds = model(batch_tensor, training=False).numpy()
        predictions = np.mean(batch_preds, axis=0)
    else:
        img_array = preprocess_image(image_bytes)
        with model_lock:
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
            predictions = model(img_tensor, training=False).numpy()[0]

    _limpiar_memoria_inferencia()

    # Si se confirma presencia de cerdo (filtro negativo activo), suprimir 'no_cerdo'
    if filter_negative and NEGATIVE_CLASS in CLASS_NAMES:
        neg_idx = CLASS_NAMES.index(NEGATIVE_CLASS)
        predictions = np.copy(predictions)
        predictions[neg_idx] = 0.0
        sum_p = np.sum(predictions)
        if sum_p > 0:
            predictions = predictions / sum_p

    latency_ms = round((perf_counter() - t0) * 1000, 2)
    top_indices = np.argsort(predictions)[::-1][:3]

    results = []
    for idx in top_indices:
        confidence = float(predictions[idx])
        is_unknown = confidence < umbral_efectivo
        class_name = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else f"clase_{idx}"

        is_no_cerdo = class_name.lower() == NEGATIVE_CLASS.lower()
        display_name = "Desconocido" if (is_unknown or is_no_cerdo) else class_name

        results.append({
            "class_id": int(idx),
            "class_name": display_name,
            "confidence": confidence,
            "confidence_pct": f"{confidence * 100:.1f}%",
            "is_unknown": is_unknown or is_no_cerdo,
            "latency_ms": latency_ms,
            "tta_applied": use_tta,
        })

    return results


def clasificar_presencia_cerdo(image_bytes: bytes, use_tta: bool = True, negative_threshold: float = 0.45) -> dict:
    """
    Confronta una imagen contra el clasificador MobileNetV2 incluyendo la clase negativa 'no_cerdo'.
    Determina si la escena corresponde a un cerdo real o a una imagen ajena (documentos, radiografías, objetos).
    """
    load_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo MobileNetV2 no cargado en memoria")

    if use_tta:
        tta_inputs = utils.preprocess_image_tta(image_bytes)
        batch_array = np.vstack([aug_array for aug_array, _label in tta_inputs])
        with model_lock:
            batch_tensor = tf.convert_to_tensor(batch_array, dtype=tf.float32)
            batch_preds = model(batch_tensor, training=False).numpy()
        predictions = np.mean(batch_preds, axis=0)
    else:
        img_array = preprocess_image(image_bytes)
        with model_lock:
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
            predictions = model(img_tensor, training=False).numpy()[0]

    _limpiar_memoria_inferencia()

    neg_idx = next(
        (i for i, c in enumerate(CLASS_NAMES) if c.lower() == NEGATIVE_CLASS.lower()),
        None,
    )
    prob_no_cerdo = float(predictions[neg_idx]) if (neg_idx is not None and neg_idx < len(predictions)) else 0.0

    top_raw_idx = int(np.argmax(predictions))
    top_raw_class = CLASS_NAMES[top_raw_idx] if 0 <= top_raw_idx < len(CLASS_NAMES) else f"clase_{top_raw_idx}"
    es_no_cerdo = (top_raw_class.lower() == NEGATIVE_CLASS.lower()) or (prob_no_cerdo >= negative_threshold)

    pig_indices = [i for i in range(len(predictions)) if i != neg_idx]
    if pig_indices:
        best_pig_idx = max(pig_indices, key=lambda i: float(predictions[i]))
        best_pig_class = CLASS_NAMES[best_pig_idx] if best_pig_idx < len(CLASS_NAMES) else f"clase_{best_pig_idx}"
        best_pig_conf = float(predictions[best_pig_idx])
    else:
        best_pig_class = None
        best_pig_conf = 0.0

    animal_identificado = best_pig_class if (not es_no_cerdo and best_pig_conf >= THRESHOLD) else None

    return {
        "es_cerdo": not es_no_cerdo,
        "es_no_cerdo": es_no_cerdo,
        "prob_no_cerdo": prob_no_cerdo,
        "prob_cerdo": round(max(0.0, 1.0 - prob_no_cerdo), 4),
        "animal_identificado": animal_identificado,
        "confianza_id": round(best_pig_conf * 100.0, 1),
        "top_raw_class": top_raw_class,
    }


def _cargar_imagen_base64_validada(image_data: str) -> tuple[bytes, Image.Image]:
    """Decodifica imagen base64, valida tamaño máximo (10MB) y la convierte a RGB."""
    if not isinstance(image_data, str) or not image_data:
        raise HTTPException(status_code=400, detail="Campo 'imagen' requerido")

    if "," in image_data:
        image_data = image_data.split(",")[1]

    try:
        image_bytes = base64.b64decode(image_data)
    except Exception:
        raise HTTPException(status_code=400, detail="Base64 inválido")

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Imagen muy grande (máx 10MB)")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image_bytes, img


def _detectar_cajas_animales_yolo(img: Image.Image) -> list:
    """Ejecuta YOLOv8 filtrando únicamente clases cuadrúpedas/porcinas plausibles."""
    yolo = load_yolo()
    boxes = []
    if not yolo:
        return boxes

    try:
        img_np = np.array(img)
        yolo_names = getattr(yolo.model, "names", {}) or {}
        is_custom_pig_yolo = len(yolo_names) <= 3
        allowed_coco_animals = {"pig", "cerdo", "cow", "sheep", "horse", "dog", "bear", "cat", "elephant"}

        with torch.inference_mode():
            yolo_res = yolo.predict(img_np, conf=0.25, verbose=False)
            if yolo_res and len(yolo_res[0].boxes) > 0:
                for b in yolo_res[0].boxes:
                    cls_id = int(b.cls[0])
                    cls_label = str(yolo_names.get(cls_id, "")).lower()
                    if is_custom_pig_yolo or cls_label in allowed_coco_animals:
                        boxes.append(b.xyxy[0].tolist())
    except Exception as e:
        logger.warning(f"Detección YOLO omitida por error: {e}")

    return boxes


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS — Reconocimiento
# ═══════════════════════════════════════════════════════════════

@app.on_event("startup")
def _precargar_modelo():
    """Precarga modelo y clases al iniciar el servicio."""
    if MODEL_PATH.exists():
        load_model()
        logger.info(f"Modelo precargado desde: {MODEL_PATH}")
    else:
        logger.warning(f"Modelo no encontrado en startup: {MODEL_PATH}")


@app.get("/")
def root():
    """Health-check simple."""
    return {
        "message": "API de Reconocimiento de Cerdas",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/ready")
def ready():
    """Healthcheck readiness — intenta cargar el modelo si no lo está."""
    load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return {"status": "ready", "modelo_cargado": True}


@app.get("/salud")
def salud():
    """
    Estado completo del servicio: telemetría de hardware, modelo, dataset y clases.

    @returns {dict} Métricas de salud, uso de RAM, CPU y estado del modelo biométrico.

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    load_model()
    telemetry = utils.get_system_telemetry()
    return {
        "status": "ok",
        "modelo_cargado": model is not None,
        "modelo_existe": MODEL_PATH.exists(),
        "modelo_ruta": str(MODEL_PATH),
        "dataset_path": DATASET_PATH,
        "dataset_existe": os.path.isdir(DATASET_PATH),
        "clases_disponibles": len(CLASS_NAMES),
        "clases": CLASS_NAMES,
        "threshold": THRESHOLD,
        "version": "2.0.0",
        "rate_limiting": "activo",
        "api_key_configured": bool(API_KEY),
        "telemetria": telemetry,
    }


@app.get("/clases")
def get_clases():
    """
    Lista todas las clases zootécnicas disponibles para clasificación biométrica.

    @returns {dict} Listado de identificadores de cerdas y umbral.

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    return {
        "clases": CLASS_NAMES,
        "total": len(CLASS_NAMES),
        "threshold": THRESHOLD
    }


@app.post("/reconocer")
@limiter.limit("60/minute")
async def reconocer(request: Request, file: UploadFile = File(...)):
    """
    Reconoce biométricamente una cerda a partir de un archivo de imagen subido (JPEG, PNG, WebP).

    @param request Petición HTTP entrante para rate limiter.
    @param file Archivo binario de imagen (UploadFile).

    @returns {dict} Predicciones Top-3 y métrica de latencia en milisegundos.

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    try:
        if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
            raise HTTPException(status_code=400, detail="Tipo de imagen no válido")

        image_bytes = await file.read()

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Imagen muy grande (máx 10MB)")

        t_start = perf_counter()
        results = predict_top3(image_bytes)
        latencia_ms = round((perf_counter() - t_start) * 1000, 2)

        return {
            "success": True,
            "resultados": results,
            "top1": results[0]["class_name"] if results else None,
            "confianza_top1": results[0]["confidence_pct"] if results else None,
            "latencia_ms": latencia_ms
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en reconocimiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reconocer_base64")
@limiter.limit("60/minute")
async def reconocer_base64(request: Request, data: dict):
    """
    Reconoce biométricamente una cerda a partir de una cadena codificada en base64.

    @param request Petición HTTP entrante para rate limiter.
    @param data Diccionario con el campo 'imagen' en base64.

    @returns {dict} Predicciones Top-3 y métrica de latencia en milisegundos.

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    try:
        image_data = data.get("imagen", "")
        if not isinstance(image_data, str) or not image_data:
            raise HTTPException(status_code=400, detail="Campo 'imagen' requerido")

        # Remueve el prefijo data:image/...;base64, si existe
        if "," in image_data:
            image_data = image_data.split(",")[1]

        try:
            image_bytes = base64.b64decode(image_data)
        except Exception:
            raise HTTPException(status_code=400, detail="Base64 inválido")

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Imagen muy grande (máx 10MB)")

        t_start = perf_counter()
        results = predict_top3(image_bytes)
        latencia_ms = round((perf_counter() - t_start) * 1000, 2)

        return {
            "success": True,
            "resultados": results,
            "top1": results[0]["class_name"] if results else None,
            "confianza_top1": results[0]["confidence_pct"] if results else None,
            "latencia_ms": latencia_ms
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en reconocimiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/estimar_biometria_base64")
@limiter.limit("60/minute")
async def estimar_biometria_base64(request: Request, data: dict):
    """
    Estima el peso corporal (kg) y la Condición Corporal (BCS 1-5) mediante morfometría óptica
    y visión artificial cenital (YOLOv8 + MobileNetV2).

    @param request Petición HTTP para control de tasa.
    @param data Diccionario con 'imagen' (base64) y opcional 'distancia_camara_cm'.

    @returns {dict} Peso estimado, BCS clasificado, identidad y métricas morfológicas.

    @author Jhon Javer Lozano Arce
    @date 25 de octubre de 2026
    """
    try:
        distancia_camara_cm = float(data.get("distancia_camara_cm", 150.0))
        t_start = perf_counter()
        image_bytes, img = _cargar_imagen_base64_validada(data.get("imagen", ""))
        img_w, img_h = img.size

        if img_w < 80 or img_h < 80:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": "RESOLUCION_INSUFICIENTE",
                    "es_cerdo": False,
                    "message": "La resolución de la imagen es insuficiente para estimar la morfometría dorsal del cerdo.",
                },
            )

        # Fase 1 obligatoria: Confrontación contra MobileNetV2 (clase 'no_cerdo' vs clases porcinas)
        clasificacion = clasificar_presencia_cerdo(image_bytes)
        if clasificacion["es_no_cerdo"]:
            prob_pct = round(clasificacion["prob_no_cerdo"] * 100.0, 1)
            conf_cerdo_pct = round(clasificacion["prob_cerdo"] * 100.0, 1)
            latencia_ms = round((perf_counter() - t_start) * 1000, 2)
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": "NO_ES_CERDO",
                    "es_cerdo": False,
                    "probabilidad_no_cerdo": prob_pct,
                    "confianza_cerdo": conf_cerdo_pct,
                    "message": (
                        f"El modelo de IA clasificó la imagen como no_cerdo ({prob_pct:.1f}% de confianza). "
                        "No se detectó un cerdo válido; sube una fotografía cenital real de un animal en corral."
                    ),
                    "latencia_ms": latencia_ms,
                },
            )

        # Fase 2: Localización ROI con YOLOv8 (solo tras confirmar presencia de cerdo)
        boxes = _detectar_cajas_animales_yolo(img)

        if boxes:
            best_box = max(boxes, key=lambda b: max(1.0, b[2] - b[0]) * max(1.0, b[3] - b[1]))
            x1, y1, x2, y2 = best_box
            box_w = max(1.0, x2 - x1)
            box_h = max(1.0, y2 - y1)
            area_pixels = box_w * box_h
            aspect_ratio = max(box_w, box_h) / max(min(box_w, box_h), 1.0)
        else:
            area_pixels = img_w * img_h * 0.65
            aspect_ratio = max(img_w, img_h) / max(min(img_w, img_h), 1.0)

        # Fase 3: Modelo morfométrico volumétrico porcino y BCS
        area_relativa = min(1.0, max(0.1, area_pixels / (img_w * img_h)))
        factor_distancia = (distancia_camara_cm / 150.0) ** 1.8
        peso_raw = 20.0 + (110.0 * area_relativa * factor_distancia)
        peso_estimado_kg = round(float(np.clip(peso_raw, 15.0, 290.0)), 2)

        # Estimación de Condición Corporal (BCS 1.0 a 5.0) por morfología elíptica dorsal
        if aspect_ratio >= 2.4:
            bcs = 2.0
        elif aspect_ratio <= 1.7:
            bcs = 4.2
        else:
            norm = (aspect_ratio - 1.7) / (2.4 - 1.7)
            bcs = 4.2 - (norm * 2.2)

        bcs = round(float(np.clip(bcs, 1.0, 5.0)), 1)

        if bcs < 2.0:
            clasificacion_bcs = "Caquéctica"
        elif bcs <= 2.7:
            clasificacion_bcs = "Delgada"
        elif bcs <= 3.4:
            clasificacion_bcs = "Ideal"
        elif bcs <= 4.1:
            clasificacion_bcs = "Sobrepeso"
        else:
            clasificacion_bcs = "Obesa"

        latencia_ms = round((perf_counter() - t_start) * 1000, 2)

        return {
            "success": True,
            "es_cerdo": True,
            "confianza_cerdo": round(clasificacion["prob_cerdo"] * 100.0, 1),
            "probabilidad_no_cerdo": round(clasificacion["prob_no_cerdo"] * 100.0, 1),
            "animal_identificado": clasificacion["animal_identificado"],
            "confianza_id": clasificacion["confianza_id"],
            "peso_estimado_kg": peso_estimado_kg,
            "condicion_corporal": bcs,
            "clasificacion_bcs": clasificacion_bcs,
            "area_pixels": int(area_pixels),
            "latencia_ms": latencia_ms,
            "modo": "ia_mobilenet_yolo",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en estimación biométrica: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analizar_celo_termico_base64")
@limiter.limit("60/minute")
async def analizar_celo_termico_base64(request: Request, data: dict):
    """
    Analiza una fotografía termográfica u óptica para HeatVision confrontando primero contra
    MobileNetV2 (clase 'no_cerdo') y estimando hiperemia/delta térmico vulvar sobre el ROI.
    """
    try:
        temp_base = float(np.clip(float(data.get("temperatura_corporal_base_c", 38.5)), 35.0, 43.0))
        t_start = perf_counter()
        image_bytes, img = _cargar_imagen_base64_validada(data.get("imagen", ""))
        img_w, img_h = img.size

        if img_w < 80 or img_h < 80:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": "RESOLUCION_INSUFICIENTE",
                    "es_cerdo": False,
                    "message": "La resolución de la imagen es insuficiente para el análisis termográfico HeatVision.",
                },
            )

        # Fase 1 obligatoria: Confrontación contra MobileNetV2 (clase 'no_cerdo' vs clases porcinas)
        clasificacion = clasificar_presencia_cerdo(image_bytes)
        if clasificacion["es_no_cerdo"]:
            prob_pct = round(clasificacion["prob_no_cerdo"] * 100.0, 1)
            conf_cerdo_pct = round(clasificacion["prob_cerdo"] * 100.0, 1)
            latencia_ms = round((perf_counter() - t_start) * 1000, 2)
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": "NO_ES_CERDO",
                    "es_cerdo": False,
                    "probabilidad_no_cerdo": prob_pct,
                    "confianza_cerdo": conf_cerdo_pct,
                    "message": (
                        f"El modelo de IA clasificó la imagen como no_cerdo ({prob_pct:.1f}% de confianza). "
                        "No se detectó una cerda válida; sube una fotografía termográfica u óptica real del animal."
                    ),
                    "latencia_ms": latencia_ms,
                },
            )

        # Fase 2: ROI animal + firma térmica / hiperemia vulvar
        boxes = _detectar_cajas_animales_yolo(img)
        img_np = np.array(img, dtype=np.float32)

        if boxes:
            best_box = max(boxes, key=lambda b: max(1.0, b[2] - b[0]) * max(1.0, b[3] - b[1]))
            x1, y1 = max(0, int(best_box[0])), max(0, int(best_box[1]))
            x2, y2 = min(img_w, int(best_box[2])), min(img_h, int(best_box[3]))
            roi = img_np[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)]
        else:
            cx1, cy1 = int(img_w * 0.2), int(img_h * 0.2)
            cx2, cy2 = max(cx1 + 1, int(img_w * 0.8)), max(cy1 + 1, int(img_h * 0.8))
            roi = img_np[cy1:cy2, cx1:cx2]

        r, g, b = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
        red_excess = float(np.mean(np.maximum(0.0, r - 0.5 * (g + b)))) / 255.0
        warm_ratio = float(np.mean((r > 170.0) & ((r - g) > 20.0) & ((r - b) > 25.0)))
        indice_hiperemia = round(float(np.clip(red_excess * 2.2 + warm_ratio * 0.65, 0.0, 1.0)), 3)

        delta_termico = round(0.2 + (indice_hiperemia * 1.6), 1)
        temperatura_vulvar_c = round(float(np.clip(temp_base + delta_termico, 37.5, 40.8)), 1)
        delta_termico = round(temperatura_vulvar_c - temp_base, 1)

        if delta_termico >= 1.2:
            hiperemia_edema_vulvar = "marcado_turgente"
            estado_celo_sugerido = "celo_activo_pico_termico"
        elif delta_termico >= 0.6:
            hiperemia_edema_vulvar = "leve"
            estado_celo_sugerido = "proestro_temprano"
        else:
            hiperemia_edema_vulvar = "ausente"
            estado_celo_sugerido = "anestro_no_celo"

        latencia_ms = round((perf_counter() - t_start) * 1000, 2)

        return {
            "success": True,
            "es_cerdo": True,
            "confianza_cerdo": round(clasificacion["prob_cerdo"] * 100.0, 1),
            "probabilidad_no_cerdo": round(clasificacion["prob_no_cerdo"] * 100.0, 1),
            "animal_identificado": clasificacion["animal_identificado"],
            "confianza_id": clasificacion["confianza_id"],
            "temperatura_vulvar_c": temperatura_vulvar_c,
            "temperatura_corporal_base_c": round(temp_base, 1),
            "delta_termico_c": delta_termico,
            "hiperemia_edema_vulvar": hiperemia_edema_vulvar,
            "estado_celo_sugerido": estado_celo_sugerido,
            "indice_hiperemia": indice_hiperemia,
            "latencia_ms": latencia_ms,
            "modo": "ia_heatvision_mobilenet",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en análisis térmico HeatVision: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analizar_audio_tos")
@limiter.limit("60/minute")
async def analizar_audio_tos(request: Request, data: dict):
    """
    Analiza acústicamente muestras de audio de galpón mediante Transformada de Fourier (FFT)
    para detectar patrones de tos porcina patológica (PRRS / Mycoplasma hyopneumoniae).

    @param request Petición HTTP entrante.
    @param data Diccionario con 'audio_base64' y opcional 'sample_rate' (defecto 16000).

    @returns {dict} Probabilidad de tos, nivel de severidad y espectrograma resumido.

    @author Jhon Javer Lozano Arce
    @date 25 de octubre de 2026
    """
    try:
        audio_b64 = data.get("audio_base64", "")
        if not isinstance(audio_b64, str) or not audio_b64:
            raise HTTPException(status_code=400, detail="Campo 'audio_base64' requerido")

        sample_rate = int(data.get("sample_rate", 16000))

        if "," in audio_b64:
            audio_b64 = audio_b64.split(",")[1]

        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="Base64 de audio inválido")

        t_start = perf_counter()

        # Conversión a señal digital (16-bit PCM o sintética si es WAV con header)
        if len(audio_bytes) > 44 and audio_bytes[:4] == b"RIFF":
            raw_signal = np.frombuffer(audio_bytes[44:], dtype=np.int16)
        else:
            raw_signal = np.frombuffer(audio_bytes, dtype=np.int16)

        if len(raw_signal) == 0:
            raw_signal = np.zeros(sample_rate, dtype=np.int16)

        signal_float = raw_signal.astype(np.float32) / 32768.0
        duracion_segundos = round(len(signal_float) / float(sample_rate), 2)

        # Análisis Espectral por FFT
        fft_vals = np.abs(np.fft.rfft(signal_float))
        fft_freqs = np.fft.rfftfreq(len(signal_float), 1.0 / sample_rate)

        # Banda de tos porcina (800 Hz a 2400 Hz) vs banda de ruido ambiental (< 400 Hz)
        mask_tos = (fft_freqs >= 800) & (fft_freqs <= 2400)
        mask_ruido = (fft_freqs < 400)

        energia_tos = float(np.sum(fft_vals[mask_tos])) if np.any(mask_tos) else 0.0
        energia_total = float(np.sum(fft_vals)) + 1e-6
        ratio_tos = min(1.0, energia_tos / energia_total)

        idx_pico = int(np.argmax(fft_vals)) if len(fft_vals) > 0 else 0
        frecuencia_pico = round(float(fft_freqs[idx_pico]), 1)

        # Algoritmo de clasificación acústica zootécnica
        if ratio_tos >= 0.40 and (900.0 <= frecuencia_pico <= 2300.0):
            probabilidad = round(float(min(0.98, 0.70 + (ratio_tos * 0.3))), 2)
            es_tos = True
            nivel_alerta = "brote_critico" if probabilidad >= 0.85 else "alerta_tos"
        elif ratio_tos >= 0.25:
            probabilidad = round(float(0.40 + (ratio_tos * 0.4)), 2)
            es_tos = False
            nivel_alerta = "alerta_tos"
        else:
            probabilidad = round(float(ratio_tos * 0.5), 2)
            es_tos = False
            nivel_alerta = "normal"

        latencia_ms = round((perf_counter() - t_start) * 1000, 2)

        return {
            "success": True,
            "probabilidad_tos": probabilidad,
            "es_tos_patologica": es_tos,
            "nivel_alerta": nivel_alerta,
            "frecuencia_pico_hz": frecuencia_pico,
            "ratio_energia_tos": round(ratio_tos, 3),
            "duracion_segundos": duracion_segundos,
            "latencia_ms": latencia_ms
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en análisis acústico de tos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ReconocerCompletoRequest(BaseModel):
    imagen: str
    incluir_deteccion: bool = True
    umbral: Optional[float] = None


@app.post("/reconocer-completo", dependencies=[Depends(verify_admin_key)])
@limiter.limit("60/minute")
async def reconocer_completo(request: Request, data: ReconocerCompletoRequest):
    """
    Pipeline Biométrico Dos Fases para Identificación de Ganado Porcino con telemetría de latencia:
    1. YOLOv8 localiza y detecta la cerda en primer plano en la escena.
    2. Recorta el ROI de la cerda para eliminar fondos y comederos que generan ruido.
    3. MobileNetV2 clasifica la cerda recortada con TTA vectorizado en batch único.

    @param request Petición HTTP para control de rate limiting.
    @param data Objeto con imagen en base64 y opciones de detección.

    @returns {dict} Resultados consolidados de YOLOv8 y MobileNetV2 con desglose de latencia y metadatos de ROI.

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    try:
        t_total_start = perf_counter()
        image_data = data.imagen
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size

        threshold = data.umbral if (data.umbral is not None and data.umbral > 0) else THRESHOLD

        detections = []
        best_crop_bytes = None
        crop_dimensions = None
        max_area = 0
        yolo_latency_ms = 0.0

        # Fase 1: Detección y localización con YOLOv8
        if data.incluir_deteccion:
            t_yolo_start = perf_counter()
            try:
                yolo = load_yolo()
                # AUD-ML-004: torch.inference_mode() deshabilita el rastreo de autograd y gradientes
                with torch.inference_mode():
                    yolo_results = yolo(img, conf=0.25, verbose=False)

                for r in yolo_results:
                    boxes = r.boxes
                    for box in boxes:
                        coords = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = yolo.model.names.get(cls_id, f"cls_{cls_id}")

                        x1, y1 = max(0, int(coords[0])), max(0, int(coords[1]))
                        x2, y2 = min(w, int(coords[2])), min(h, int(coords[3]))
                        area = (x2 - x1) * (y2 - y1)

                        detections.append({
                            "clase": label,
                            "confianza": round(conf * 100),
                            "bbox": [x1, y1, x2, y2]
                        })

                        # Seleccionar la cerda con mayor área en primer plano
                        if area > max_area and (x2 - x1) > 40 and (y2 - y1) > 40:
                            max_area = area
                            crop_w, crop_h = (x2 - x1), (y2 - y1)
                            crop_dimensions = [crop_w, crop_h]
                            crop_img = img.crop((x1, y1, x2, y2))
                            buf = io.BytesIO()
                            crop_img.save(buf, format="JPEG")
                            best_crop_bytes = buf.getvalue()

            except Exception as yolo_err:
                logger.warning(f"Error no fatal en detección YOLO: {yolo_err}")
            yolo_latency_ms = round((perf_counter() - t_yolo_start) * 1000, 2)

        # Fase 2: Clasificación MobileNetV2 sobre el ROI recortado
        t_tf_start = perf_counter()
        target_bytes = best_crop_bytes if best_crop_bytes is not None else image_bytes
        # Si YOLO detectó un cerdo o se recortó el ROI, suprimir la clase 'no_cerdo'
        filtro_no_cerdo = (best_crop_bytes is not None) or (len(detections) > 0)
        tf_results = predict_top3(target_bytes, use_tta=True, threshold=threshold, filter_negative=filtro_no_cerdo)
        mobilenet_latency_ms = round((perf_counter() - t_tf_start) * 1000, 2)
        total_latency_ms = round((perf_counter() - t_total_start) * 1000, 2)

        top1 = tf_results[0]["class_name"] if tf_results else "Desconocido"

        return {
            "success": True,
            "identified_as": top1,
            "confidence_top1": tf_results[0]["confidence_pct"] if tf_results else "0%",
            "confidence_score": tf_results[0]["confidence"] if tf_results else 0.0,
            "tensorflow_predictions": tf_results,
            "yolo_detections": detections,
            "total_detections": len(detections),
            "roi_cropped": best_crop_bytes is not None,
            "roi_dimensions": crop_dimensions,
            "threshold_applied": threshold,
            "latencia_total_ms": total_latency_ms,
            "latencia_yolo_ms": yolo_latency_ms,
            "latencia_mobilenet_ms": mobilenet_latency_ms
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error crítico en pipeline biométrico: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class ConfirmacionRequest(BaseModel):
    class_name: str
    confidence: float
    animal_id: Optional[int] = None
    foto_base64: Optional[str] = None
    clase_confirmada: Optional[str] = None
    usuario_id: Optional[int] = None


@app.post("/confirmar")
async def confirmar(data: ConfirmacionRequest):
    """
    Registra la confirmación del usuario sobre una predicción.
    Se usa para retroalimentación y mejora del modelo.
    """
    try:
        clase_confirmada = data.clase_confirmada or data.class_name

        # Persist confirmation to log
        confirm_log_path = BASE_DIR / "confirmaciones_log.jsonl"
        entry = {
            "timestamp": datetime.now().isoformat(),
            "clase_predicha": data.class_name,
            "clase_confirmada": clase_confirmada,
            "confidence": data.confidence,
            "animal_id": data.animal_id,
            "usuario_id": data.usuario_id,
        }
        with open(confirm_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Save image to dataset if provided
        if data.foto_base64:
            try:
                image_data = data.foto_base64
                if "," in image_data:
                    image_data = image_data.split(",")[1]
                image_bytes = base64.b64decode(image_data)

                class_dir = os.path.join(RECONOCIMIENTOS_PATH, clase_confirmada)
                os.makedirs(class_dir, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                uuid_str = base64.urlsafe_b64encode(os.urandom(6)).decode().rstrip("=")
                filename = f"confirm_{timestamp}_{uuid_str}.jpg"
                filepath = os.path.join(class_dir, filename)

                with open(filepath, "wb") as f:
                    f.write(image_bytes)

                entry["imagen_guardada"] = filename
            except Exception as img_err:
                logger.error(f"Error guardando imagen de confirmacion: {img_err}")

        # Count confirmations to trigger retrain if enough
        try:
            with open(confirm_log_path) as _cl:
                total_confirmaciones = sum(1 for _ in _cl)
                entry["total_confirmaciones"] = total_confirmaciones
        except Exception:
            pass

        return {
            "success": True,
            "message": "Confirmación registrada",
            "data": entry
        }
    except Exception as e:
        logger.error(f"Error en confirmar: {e}")
        raise HTTPException(status_code=500, detail=str(e))

 
# ═══════════════════════════════════════════════════════════════
# ENDPOINTS — YOLOv8 (Detección de objetos)
# ═══════════════════════════════════════════════════════════════

# Busca automáticamente el peso YOLO en distintas ubicaciones posibles
yolo_model_paths = [
    BASE_DIR / "runs" / "detect" / "output" / "yolo_detector" / "train" / "weights" / "best.pt",
    BASE_DIR / "runs" / "detect" / "output" / "yolo_detector" / "runs" / "weights" / "best.pt",
    BASE_DIR / "yolov8n.pt",
]
YOLO_MODEL_PATH = next((str(p) for p in yolo_model_paths if p.exists()), str(yolo_model_paths[-1]))
yolo_model = None


def load_yolo():
    """Carga el modelo YOLO con lazy loading."""
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        logger.info(f"YOLO cargado: {yolo_model.model.names}")
    return yolo_model


@app.post("/detectar")
async def detectar(file: UploadFile = File(...)):
    """
    Detecta cerdas en una imagen usando YOLOv8.
    Retorna bounding boxes, clase y confianza por detección.
    """
    try:
        if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
            raise HTTPException(status_code=400, detail="Tipo de imagen no válido")

        yolo = load_yolo()

        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))

        # AUD-ML-004: Inferencia en modo estricto sin retención de tensores
        with torch.inference_mode():
            results = yolo(img, conf=0.25, verbose=False)

        _limpiar_memoria_inferencia()

        deteccions = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                deteccions.append({
                    "clase": yolo.model.names[int(box.cls[0])],
                    "confianza": round(float(box.conf[0]) * 100),
                    "bbox": box.xyxy[0].tolist()
                })

        return {
            "success": True,
            "detecciones": len(deteccions),
            "resultados": deteccions
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en detección: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/detector/status")
async def detector_status():
    """Estado y configuración del detector YOLO."""
    yolo = load_yolo()
    return {
        "success": True,
        "modelo": "YOLOv8",
        "clases": yolo.model.names,
        "peso": YOLO_MODEL_PATH
    }


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS — Detección de Enfermedades
# ═══════════════════════════════════════════════════════════════

class EnfermedadRequest(BaseModel):
    animal_id: str
    peso_actual_kg: float
    peso_anterior_kg: float
    dias_entre_pesajes: int
    edad_dias: Optional[int] = None
    temperatura_opcional: Optional[float] = None


@app.post("/detectar-enfermedad")
async def detectar_enfermedad(data: EnfermedadRequest):
    """
    Evalúa riesgo de enfermedad basado en cambios de peso y temperatura.
    Reglas:
      - Pérdida > 5% en 7 días → ALTO riesgo
      - Pérdida > 3% en 7 días → MEDIO riesgo
      - Sin ganancia en joven (<90 días) → MEDIO riesgo
      - Normal → BAJO riesgo
    """
    try:
        peso_actual = data.peso_actual_kg
        peso_anterior = data.peso_anterior_kg
        dias = data.dias_entre_pesajes
        edad = data.edad_dias
        temperatura = data.temperatura_opcional

        factores = []
        riesgo = "BAJO"
        alerta = False
        cambio_pct = None

        if peso_anterior > 0 and dias > 0:
            cambio_pct = ((peso_actual - peso_anterior) / peso_anterior) * 100
            perdida_diaria = cambio_pct / dias if dias > 0 else 0

            perdida_pct_7d = perdida_diaria * 7

            if perdida_pct_7d < -5:
                riesgo = "ALTO"
                alerta = True
                factores.append(f"Pérdida de peso severa: {cambio_pct:+.1f}% en {dias} días")
            elif perdida_pct_7d < -3:
                riesgo = "MEDIO"
                factores.append(f"Pérdida de peso moderada: {cambio_pct:+.1f}% en {dias} días")
            elif edad is not None and edad < 90 and cambio_pct <= 0:
                riesgo = "MEDIO"
                factores.append(f"Animal joven ({edad} días) sin ganancia de peso")
            else:
                factores.append(f"Peso estable o ganancia: {cambio_pct:+.1f}% en {dias} días")
        else:
            factores.append("Datos insuficientes para evaluar cambio de peso")

        if temperatura is not None:
            if temperatura >= 40:
                riesgo = "ALTO"
                alerta = True
                factores.append(f"Fiebre detectada: {temperatura}°C")
            elif temperatura >= 39.5:
                if riesgo != "ALTO":
                    riesgo = "MEDIO"
                factores.append(f"Temperatura elevada: {temperatura}°C")
            else:
                factores.append(f"Temperatura normal: {temperatura}°C")

        recomendaciones = {
            "ALTO": "Aislar al animal, notificar al veterinario de inmediato, realizar revisión clínica completa, considerar tomar muestra para laboratorio.",
            "MEDIO": "Monitorear cada 6 horas, registrar temperatura y peso diariamente, evaluar apetito y comportamiento, preparar plan de intervención.",
            "BAJO": "Continuar monitoreo rutinario, mantener programa de alimentación y manejo sanitario habitual.",
        }

        return {
            "success": True,
            "animal_id": data.animal_id,
            "riesgo": riesgo,
            "factores": factores,
            "recomendacion": recomendaciones.get(riesgo, ""),
            "alerta": alerta,
            "cambio_peso_pct": round(cambio_pct, 2) if peso_anterior > 0 and dias > 0 else None,
        }

    except Exception as e:
        logger.error(f"Error en detección de enfermedad: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS — Dataset
# ═══════════════════════════════════════════════════════════════

class ExportarFotoRequest(BaseModel):
    class_name: str
    image_base64: str


@app.post("/exportar-foto")
@limiter.limit("10/minute")
async def exportar_foto(request: Request, data: ExportarFotoRequest, auth: bool = Depends(verify_admin_key)):
    """
    Guarda una foto directamente en el dataset de entrenamiento.
    Ruta final: {DATASET_PATH}/{class_name}/foto.jpg
    """
    try:
        class_name = data.class_name.strip()
        if not class_name:
            raise HTTPException(status_code=400, detail="class_name es requerido")
        if not re.match(r'^[a-zA-Z0-9_\-]+$', class_name):
            raise HTTPException(status_code=400, detail="class_name contiene caracteres inválidos")

        class_dir = os.path.join(DATASET_PATH, class_name)
        os.makedirs(class_dir, exist_ok=True)

        image_data = data.image_base64
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{class_name}_{timestamp}.jpg"
        filepath = os.path.join(class_dir, filename)

        with open(filepath, "wb") as f:
            f.write(image_bytes)

        return {"success": True, "filename": filename, "class_name": class_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dataset-estadisticas")
async def dataset_estadisticas():
    """
    Retorna estadísticas del dataset:
    - Lista de clases con conteo de imágenes
    - Total global de imágenes y clases
    """
    global _stats_cache
    now = time()
    if _stats_cache["data"] is not None and (now - _stats_cache["timestamp"]) < CACHE_TTL:
        return _stats_cache["data"]

    try:
        if not os.path.exists(DATASET_PATH):
            data = {"clases": [], "total_global": 0, "total_clases": 0}
            _stats_cache = {"data": data, "timestamp": now}
            return data

        clases = []
        total_global = 0
        for d in sorted(os.listdir(DATASET_PATH)):
            dir_path = os.path.join(DATASET_PATH, d)
            if os.path.isdir(dir_path):
                count = len([f for f in os.listdir(dir_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
                clases.append({"nombre": d, "total": count})
                total_global += count

        data = {
            "clases": clases,
            "total_global": total_global,
            "total_clases": len(clases),
        }
        _stats_cache = {"data": data, "timestamp": now}
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dataset/diagnostico")
async def dataset_diagnostico():
    """
    Diagnóstico zootécnico del dataset de imágenes de cerdas:
    - Conteo por animal y detección de desbalance extremo.
    - Identificación de clases críticas con < 30 imágenes (muestra insuficiente).
    - Cálculo de ratio de desbalance (max / min) y recomendaciones de captura.

    @returns {dict} Diagnóstico detallado con alertas de balanceo zootécnico y salud de dataset.

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    try:
        if not os.path.exists(DATASET_PATH):
            return {
                "success": False,
                "error": "DATASET_PATH no existe en el sistema",
                "dataset_path": DATASET_PATH,
                "total_imagenes": 0,
                "total_clases": 0
            }

        conteo_por_clase = {}
        for d in sorted(os.listdir(DATASET_PATH)):
            dir_path = os.path.join(DATASET_PATH, d)
            if os.path.isdir(dir_path):
                imgs = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                conteo_por_clase[d] = len(imgs)

        total_imagenes = sum(conteo_por_clase.values())
        total_clases = len(conteo_por_clase)

        if total_clases == 0:
            return {
                "success": True,
                "total_imagenes": 0,
                "total_clases": 0,
                "mensaje": "Dataset vacío"
            }

        counts = list(conteo_por_clase.values())
        min_imgs = min(counts)
        max_imgs = max(counts)
        ratio_desbalance = round(max_imgs / max(min_imgs, 1), 2)
        promedio_por_clase = round(total_imagenes / total_clases, 1)

        clases_criticas = [c for c, cnt in conteo_por_clase.items() if cnt < 30 and c != NEGATIVE_CLASS]
        clases_optimas = [c for c, cnt in conteo_por_clase.items() if cnt >= 80]

        recomendaciones = []
        if clases_criticas:
            recomendaciones.append(
                f"Capturar urgentemente más fotografías para {len(clases_criticas)} cerdas con menos de 30 fotos: {', '.join(clases_criticas[:5])}"
            )
        if ratio_desbalance > 5.0:
            recomendaciones.append(
                f"Ratio de desbalance elevado ({ratio_desbalance}:1). Se recomienda aplicar oversampling o balanceo ponderado al entrenar."
            )
        if not recomendaciones:
            recomendaciones.append("Dataset balanceado con representatividad fotográfica zootécnica adecuada.")

        return {
            "success": True,
            "total_imagenes": total_imagenes,
            "total_clases": total_clases,
            "promedio_por_clase": promedio_por_clase,
            "min_imagenes": min_imgs,
            "max_imagenes": max_imgs,
            "ratio_desbalance": ratio_desbalance,
            "alerta_desbalance": ratio_desbalance > 3.0,
            "clases_criticas": clases_criticas,
            "total_criticas": len(clases_criticas),
            "clases_optimas": clases_optimas,
            "distribucion": conteo_por_clase,
            "recomendaciones_zootecnicas": recomendaciones
        }
    except Exception as e:
        logger.error(f"Error en diagnostico del dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training-stats")
async def training_stats():
    """
    Retorna estadísticas para el pipeline de reentrenamiento:
    - Total de fotos en la carpeta fuente (imagenes_originales)
    - Conteo por clase
    """
    try:
        source_path = r"D:\cerdos\imagenes_originales"
        if not os.path.exists(source_path):
            return {
                "source_path": source_path,
                "total_source_photos": 0,
                "classes": [],
                "error": None
            }

        classes_data = []
        total = 0
        for d in sorted(os.listdir(source_path)):
            dir_path = os.path.join(source_path, d)
            if os.path.isdir(dir_path):
                count = len([f for f in os.listdir(dir_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
                classes_data.append({"class": d, "count": count})
                total += count

        return {
            "source_path": source_path,
            "total_source_photos": total,
            "classes": classes_data,
            "error": None
        }
    except Exception as e:
        return {
            "source_path": None,
            "total_source_photos": 0,
            "classes": [],
            "error": str(e)
        }


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS — Entrenamiento
# ═══════════════════════════════════════════════════════════════

class EntrenarRequest(BaseModel):
    include_classes: Optional[List[str]] = None
    exclude_classes: Optional[List[str]] = None
    config: Optional[dict] = None


@app.get("/ml/training-config")
async def training_config():
    """
    Retorna la configuración por defecto del script de entrenamiento.
    Usada por el frontend para llenar el panel de configuración avanzada.
    """
    return {
        "batch_size": 16,
        "epochs_frozen": 60,
        "epochs_finetune": 80,
        "lr": 0.0001,
        "finetune_lr": 0.000001,
        "max_images_per_class": 200,
        "max_no_cerdo": 80,
        "unfreeze_layers": 8,
        "disable_mixup": False,
        "mixup_alpha": 0.2,
        "oversample_min": 80,
        "focal_gamma": 1.5,
        "disable_class_weights": False,
    }


@app.post("/entrenar")
@limiter.limit("10/minute")
async def iniciar_entrenamiento(request: Request, data: EntrenarRequest = None, auth: bool = Depends(verify_admin_key)):
    """
    Inicia entrenamiento asíncrono del modelo MobileNetV2.
    - Crea backup automático del modelo actual
    - Delega a training_manager que corre en un thread separado
    - Opcional: include_classes para entrenar solo clases específicas
    - Opcional: config con hiperparámetros avanzados
    """
    try:
        backup_current_model()

        include = data.include_classes if data else None
        exclude = data.exclude_classes if data else None
        config = data.config if data else None

        task_id = manager.start_training(include_classes=include, exclude_classes=exclude, config=config)
        return {"success": True, "task_id": task_id, "message": "Entrenamiento iniciado"}
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/entrenar/historial")
async def historial_entrenamiento():
    """Historial completo de entrenamientos realizados."""
    tasks = manager.get_all_tasks()
    return {"success": True, "tasks": tasks}


@app.get("/entrenar/{task_id}")
async def estado_entrenamiento(task_id: str):
    """Consulta el estado actual de una tarea de entrenamiento."""
    status = manager.get_status(task_id)
    if status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Tarea no encontrada")

    # Calcular elapsed_seconds si started_at está presente
    started_at = status.get("started_at")
    if started_at:
        try:
            start_dt = datetime.fromisoformat(started_at)
            elapsed = (datetime.now() - start_dt).total_seconds()
            status["elapsed_seconds"] = int(elapsed)
        except Exception:
            pass

    return {"success": True, "task": status}


@app.get("/entrenar/{task_id}/log")
async def obtener_log_entrenamiento(task_id: str):
    """Retorna el log de error completo de una tarea de entrenamiento."""
    logs_dir = BASE_DIR / "logs"
    log_path = logs_dir / f"train_{task_id}_error.log"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log no disponible")
    return {"success": True, "log": log_path.read_text(encoding="utf-8")}


@app.get("/entrenar/{task_id}/history")
async def obtener_history_entrenamiento(task_id: str):
    """Retorna el historial de métricas por época (phase1 + phase2)."""
    task_dir = TASKS_DIR / task_id
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    history = {}
    for phase_file in ["history_phase1.json", "history_phase2.json"]:
        path = task_dir / phase_file
        if path.exists():
            phase = phase_file.replace("history_", "").replace(".json", "")
            history[phase] = json.loads(path.read_text(encoding="utf-8"))
    return {"success": True, "history": history}


@app.post("/entrenar/{task_id}/cancel")
async def cancelar_entrenamiento(task_id: str, auth: bool = Depends(verify_admin_key)):
    """Cancela una tarea de entrenamiento en ejecución."""
    ok = manager.cancel_training(task_id)
    if not ok:
        raise HTTPException(status_code=400, detail="No se puede cancelar la tarea (no encontrada o ya finalizada)")
    return {"success": True, "message": "Entrenamiento cancelado"}


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS — Gestión del Modelo
# ═══════════════════════════════════════════════════════════════

@app.post("/modelo/recargar")
@limiter.limit("10/minute")
async def recargar_modelo(request: Request, auth: bool = Depends(verify_admin_key)):
    """
    Recarga el modelo desde disco.
    Útil después de un entrenamiento para usar el nuevo modelo sin reiniciar el servicio.
    """
    global model, CLASS_NAMES, THRESHOLD
    try:
        with model_lock:
            model = None
            CLASS_NAMES = load_classes()
            _reload_threshold()
            load_model()
        return {"success": True, "message": "Modelo recargado exitosamente", "clases": len(CLASS_NAMES)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/modelo/versiones")
async def modelo_versiones():
    """Lista todas las versiones de backup disponibles para restauración."""
    return {
        "success": True,
        "backups": list_model_backups(),
        "actual": MODEL_PATH.name if MODEL_PATH.exists() else None
    }


@app.get("/modelos/info")
async def modelos_info():
    """
    Retorna información técnica y metadatos del modelo biométrico en producción.

    Expone arquitectura de entrada, número de clases zootécnicas, tamaño del archivo de pesos,
    umbral de corte y los registros históricos de entrenamiento.

    @returns {dict} Metadatos completos del modelo de clasificación y detector.

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    registry_file = BASE_DIR / "model_registry.json"
    registry_entries = []
    if registry_file.exists():
        try:
            with open(registry_file, encoding="utf-8") as rf:
                registry_entries = json.load(rf)
        except Exception:
            registry_entries = []

    model_stats = {}
    if MODEL_PATH.exists():
        stat = MODEL_PATH.stat()
        model_stats = {
            "archivo": MODEL_PATH.name,
            "tamano_mb": round(stat.st_size / (1024 * 1024), 2),
            "ultima_modificacion": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

    yolo_path = BASE_DIR / "yolov8n.pt"
    yolo_stats = {}
    if yolo_path.exists():
        ystat = yolo_path.stat()
        yolo_stats = {
            "archivo": yolo_path.name,
            "tamano_mb": round(ystat.st_size / (1024 * 1024), 2),
            "existe": True,
        }
    else:
        yolo_stats = {"existe": False}

    return {
        "success": True,
        "modelo_clasificador": {
            "modelo_cargado": model is not None,
            "modelo_existe": MODEL_PATH.exists(),
            "detalles_archivo": model_stats,
            "dimensiones_entrada": list(IMG_SIZE),
            "total_clases": len(CLASS_NAMES),
            "clases": CLASS_NAMES,
            "umbral_activo": THRESHOLD,
            "version": "2.0.0",
        },
        "detector_yolo": yolo_stats,
        "historial_registro": registry_entries[-5:],
        "total_versiones_registradas": len(registry_entries),
    }


class RestaurarRequest(BaseModel):
    version: str


@app.post("/modelo/restaurar")
@limiter.limit("10/minute")
async def modelo_restaurar(request: Request, data: RestaurarRequest, auth: bool = Depends(verify_admin_key)):
    """
    Restaura un modelo desde un backup existente.
    - Crea backup del modelo actual antes de restaurar
    - Recarga automáticamente el modelo restaurado
    """
    global model, CLASS_NAMES
    try:
        backup_file = MODELS_BACKUP_DIR / f"modelo_{data.version}.h5"
        if not backup_file.exists():
            raise HTTPException(status_code=404, detail=f"Version {data.version} no encontrada")

        backup_current_model()

        shutil.copy2(str(backup_file), str(MODEL_PATH))
        classes_file = MODELS_BACKUP_DIR / f"classes_{data.version}.json"
        if classes_file.exists():
            shutil.copy2(str(classes_file), str(BASE_DIR / "classes.json"))

        with model_lock:
            model = None
            CLASS_NAMES = load_classes()
            load_model()

        return {
            "success": True,
            "message": f"Modelo restaurado a version {data.version}",
            "clases": len(CLASS_NAMES),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test")
async def test_endpoint(data: dict):
    """Endpoint de prueba para verificar conectividad."""
    return {
        "success": True,
        "message": "Test exitoso",
        "data": data
    }


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS — Farrowing Monitor (Detección de Partos con IA)
# ═══════════════════════════════════════════════════════════════

class FarrowingAnalisisRequest(BaseModel):
    camara_id: str = "cam-01"
    tiempo_en_postura_min: int = 15
    dias_gestacion: int = 114
    lechones_previos: int = 0
    image_base64: str


@app.get("/farrowing/camaras-config")
def farrowing_camaras_config():
    """
    Retorna la configuración predeterminada de cámaras para jaulas de maternidad.

    @returns {dict} Lista de cámaras de farrowing y estado de conexión.

    @author Jhon Javer Lozano Arce

    @date 26 de septiembre de 2026
    """
    return {
        "status": "success",
        "camaras": CAMARAS_DEFAULT,
        "total": len(CAMARAS_DEFAULT)
    }


@app.post("/farrowing/analizar")
async def farrowing_analizar_upload(
    file: UploadFile = File(...),
    camara_id: str = "cam-01",
    tiempo_en_postura_min: int = 15,
    dias_gestacion: int = 114,
    lechones_previos: int = 0
):
    """
    Analiza un frame fotográfico subido por multipart para detección de parto y posturas.

    @param file Imagen binaria del frame capturado por la cámara.
    @param camara_id Identificador de la cámara paritoria.
    @param tiempo_en_postura_min Minutos continuos en la misma postura.
    @param dias_gestacion Días de preñez de la cerda.
    @param lechones_previos Conteo acumulado de lechones en la jaula.

    @returns {dict} Análisis zootécnico de farrowing y alertas en tiempo real.

    @author Jhon Javer Lozano Arce

    @date 26 de septiembre de 2026
    """
    image_bytes = await file.read()
    resultado = farrowing_analyzer.analizar_imagen(
        image_bytes=image_bytes,
        camara_id=camara_id,
        tiempo_en_postura_min=tiempo_en_postura_min,
        dias_gestacion=dias_gestacion,
        lechones_previos=lechones_previos
    )
    return {"status": "success", "data": resultado}


@app.post("/farrowing/analizar-base64")
def farrowing_analizar_base64(data: FarrowingAnalisisRequest):
    """
    Analiza un frame codificado en base64 para detección de parto y posturas.

    @param data Objeto con imagen base64 y parámetros zootécnicos.

    @returns {dict} Análisis zootécnico de farrowing y alertas en tiempo real.

    @author Jhon Javer Lozano Arce

    @date 26 de septiembre de 2026
    """
    try:
        raw_b64 = re.sub(r"^data:image/.+;base64,", "", data.image_base64)
        image_bytes = base64.b64decode(raw_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Base64 inválido: {str(e)}")

    resultado = farrowing_analyzer.analizar_imagen(
        image_bytes=image_bytes,
        camara_id=data.camara_id,
        tiempo_en_postura_min=data.tiempo_en_postura_min,
        dias_gestacion=data.dias_gestacion,
        lechones_previos=data.lechones_previos
    )
    return {"status": "success", "data": resultado}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
