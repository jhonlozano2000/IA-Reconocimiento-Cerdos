"""
utils.py — Shared utilities for the Porci-Integral AI module.

Consolidates duplicated code across app_fastapi.py, entrenar_v2.py,
cctv_monitor.py, and training_manager.py.
"""

import os
import json
import base64
import logging
import time
from pathlib import Path
from io import BytesIO
from typing import Optional, Dict, Any, List

import numpy as np
from PIL import Image
import psutil


BASE_DIR = Path(__file__).resolve().parent
IMG_SIZE = (224, 224)
DEFAULT_THRESHOLD = 0.50
NEGATIVE_CLASS = "no_cerdo"  # folder name for negative/non-pig samples


def load_env(env_path: Optional[Path] = None) -> None:
    """
    Carga las variables de entorno desde el archivo .env de forma idempotente.

    @param env_path Ruta opcional al archivo .env.

    @returns {None}

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    env_path = env_path or BASE_DIR / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                if _k.strip() not in os.environ:
                    os.environ[_k.strip()] = _v.strip().strip('"').strip("'")


def load_classes(classes_path: Optional[Path] = None) -> list:
    """
    Obtiene la lista autoritativa de clases zootécnicas desde classes.json o escaneo de carpetas.

    @param classes_path Ruta opcional al archivo classes.json.

    @returns {list} Lista con nombres de clases registradas.

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    classes_path = classes_path or BASE_DIR / "classes.json"
    if classes_path.exists():
        with open(classes_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("classes", [])
    dataset_path = os.environ.get("DATASET_PATH", "")
    if dataset_path and os.path.exists(dataset_path):
        return sorted([
            d for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ])
    return []


def load_threshold(classes_path: Optional[Path] = None) -> float:
    """
    Obtiene el umbral mínimo de confianza configurado para clasificación biométrica.

    @param classes_path Ruta opcional al archivo classes.json.

    @returns {float} Valor numérico del umbral de confianza (ej. 0.50).

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    classes_path = classes_path or BASE_DIR / "classes.json"
    if classes_path.exists():
        with open(classes_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("threshold", DEFAULT_THRESHOLD)
    return DEFAULT_THRESHOLD


def preprocess_image(image_bytes: bytes, img_size=IMG_SIZE) -> np.ndarray:
    """
    Transforma bytes de imagen en un tensor NumPy normalizado (1, 224, 224, 3) para MobileNetV2.

    @param image_bytes Contenido binario de la imagen.
    @param img_size Tupla con dimensiones objetivo (ancho, alto).

    @returns {np.ndarray} Tensor normalizado en rango [0, 1].

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize(img_size, Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def preprocess_image_tta(image_bytes: bytes, img_size=IMG_SIZE) -> list:
    """
    Genera variantes espaciales con Test-Time Augmentation (TTA) para ensamble en inferencia.

    @param image_bytes Contenido binario de la imagen.
    @param img_size Dimensiones objetivo de redimensionamiento.

    @returns {list} Lista de tuplas (tensor, etiqueta_transformación).

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    def _to_array(pil_img):
        arr = pil_img.resize(img_size, Image.Resampling.BILINEAR)
        arr = np.array(arr, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    versions = [
        (img.copy(), "original"),
        (img.transpose(Image.FLIP_LEFT_RIGHT), "flip"),
    ]

    # Rotación +10 grados
    rot_p = img.rotate(10, resample=Image.Resampling.BILINEAR, expand=False, fillcolor="black")
    versions.append((rot_p, "rot+10"))

    # Rotación -10 grados
    rot_n = img.rotate(-10, resample=Image.Resampling.BILINEAR, expand=False, fillcolor="black")
    versions.append((rot_n, "rot-10"))

    return [(_to_array(v), label) for v, label in versions]


def predict_with_tta(model, image_bytes: bytes, class_names: list, threshold: float = 0.50) -> Dict[str, Any]:
    """
    Ejecuta inferencia con Test-Time Augmentation (TTA) vectorizado en batch único y mide latencia.

    @param model Modelo MobileNetV2 de Keras cargado en memoria.
    @param image_bytes Bytes en crudo de la imagen del animal.
    @param class_names Lista ordenada de clases biométricas registradas.
    @param threshold Umbral mínimo de confianza zootécnica (defecto 0.50).

    @returns {dict} Diccionario con class_name, confidence, confidence_pct, class_id y latency_ms.

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    t0 = time.perf_counter()
    tta_inputs = preprocess_image_tta(image_bytes)
    
    # Vectorización batch de tamaño (4, 224, 224, 3) para ejecución única en CPU
    batch_array = np.vstack([aug_array for aug_array, _label in tta_inputs])
    batch_preds = model.predict(batch_array, verbose=0)

    avg_preds = np.mean(batch_preds, axis=0)
    top_idx = int(np.argmax(avg_preds))
    confidence = float(avg_preds[top_idx])
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    if confidence < threshold:
        class_name = "Desconocido"
    elif 0 <= top_idx < len(class_names) and class_names[top_idx].lower() == NEGATIVE_CLASS.lower():
        class_name = "Desconocido"
    elif 0 <= top_idx < len(class_names):
        class_name = class_names[top_idx]
    else:
        class_name = f"clase_{top_idx}"

    return {
        "class_name": class_name,
        "confidence": confidence,
        "confidence_pct": f"{confidence * 100:.1f}%",
        "class_id": top_idx,
        "latency_ms": latency_ms,
        "tta_applied": True,
    }


def decode_base64_image(image_data: str) -> bytes:
    """
    Decodifica una cadena base64 removiendo prefijos de Data URI MIME si existen.

    @param image_data Cadena en formato base64 con o sin prefijo data:image.

    @returns {bytes} Contenido binario decodificado.

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    if "," in image_data:
        image_data = image_data.split(",")[1]
    return base64.b64decode(image_data)


def get_system_telemetry() -> Dict[str, Any]:
    """
    Obtiene métricas de telemetría de hardware (CPU, RAM y núcleos lógicos) para observabilidad MLOps.

    @returns {dict} Estadísticas de recursos del sistema anfitrión.

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    vmem = psutil.virtual_memory()
    return {
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_total_mb": round(vmem.total / (1024 * 1024), 1),
        "ram_available_mb": round(vmem.available / (1024 * 1024), 1),
        "ram_used_pct": vmem.percent,
    }


def setup_logging(name: str = __name__, level=logging.INFO) -> logging.Logger:
    """
    Configura y retorna el logger corporativo estándar del módulo de Inteligencia Artificial.

    @param name Nombre identificador del logger.
    @param level Nivel mínimo de severidad de logging.

    @returns {logging.Logger} Instancia configurada de Logger.

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger(name)
