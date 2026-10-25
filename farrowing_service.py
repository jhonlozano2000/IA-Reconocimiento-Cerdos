"""
farrowing_service.py — Motor de IA para Detección de Partos, Posturas y Eventos en Maternidad
=============================================================================================
Analiza frames de video de jaulas paritorias para detectar:
- Posturas de la cerda (Decúbito lateral, Decúbito esternal, De pie)
- Presencia y conteo de lechones neonatos
- Alertas zootécnicas críticas (Parto inminente, Parto activo, Distocia, Aplastamiento)
"""

import io
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
    HAVE_YOLO = True
except ImportError:
    HAVE_YOLO = False

import utils

logger = logging.getLogger(__name__)

# Configuración de cámaras de maternidad predeterminadas
CAMARAS_DEFAULT = [
    {"id": "cam-01", "name": "Cama 01 - Paritorio Norte", "jaula": "J-01", "stream_url": ""},
    {"id": "cam-02", "name": "Cama 02 - Paritorio Norte", "jaula": "J-02", "stream_url": ""},
    {"id": "cam-03", "name": "Cama 03 - Paritorio Norte", "jaula": "J-03", "stream_url": ""},
    {"id": "cam-04", "name": "Cama 04 - Paritorio Sur", "jaula": "J-04", "stream_url": ""},
    {"id": "cam-05", "name": "Cama 05 - Paritorio Sur", "jaula": "J-05", "stream_url": ""},
    {"id": "cam-06", "name": "Cama 06 - Paritorio Sur", "jaula": "J-06", "stream_url": ""},
    {"id": "cam-07", "name": "Cama 07 - Paritorio Centro", "jaula": "J-07", "stream_url": ""},
    {"id": "cam-08", "name": "Cama 08 - Paritorio Centro", "jaula": "J-08", "stream_url": ""},
]


class FarrowingAnalyzer:
    """
    Analizador biométrico de parición para jaulas de maternidad porcina.
    """

    def __init__(self, yolo_model_path: Optional[str] = None):
        """
        Inicializa el modelo de detección de objetos YOLO para farrowing.
        """
        self.yolo_model = None
        if HAVE_YOLO:
            try:
                model_file = yolo_model_path or (utils.BASE_DIR / "yolov8n.pt")
                if str(model_file).endswith(".pt") and utils.BASE_DIR.exists():
                    self.yolo_model = YOLO(str(model_file))
                    logger.info(f"YOLO detector cargado para Farrowing desde: {model_file}")
            except Exception as e:
                logger.warning(f"No se pudo inicializar YOLO en FarrowingAnalyzer: {e}")

    def analizar_imagen(
        self,
        image_bytes: bytes,
        camara_id: str = "cam-01",
        tiempo_en_postura_min: int = 15,
        dias_gestacion: int = 114,
        lechones_previos: int = 0
    ) -> Dict[str, Any]:
        """
        Analiza un frame de video de jaula paritoria y genera un dictamen zootécnico en tiempo real.

        @param image_bytes Contenido binario de la imagen del frame.
        @param camara_id Identificador de la cámara paritoria.
        @param tiempo_en_postura_min Minutos acumulados observados en la postura actual.
        @param dias_gestacion Días transcurridos desde el servicio/inseminación de la cerda.
        @param lechones_previos Conteo previo registrado de lechones en la camada.

        @returns {dict} Resultados biométricos, postura estimada, probabilidad y alertas.

        @author Jhon Javer Lozano Arce

        @date 26 de septiembre de 2026
        """
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            width, height = image.size
        except Exception as e:
            logger.error(f"Error al decodificar imagen para farrowing: {e}")
            return self._respuesta_fallback(camara_id, error=f"Imagen inválida: {str(e)}")

        postura = "decubito_esternal"
        cajas_detectadas = []
        lechones_detectados = lechones_previos

        # 1. Inferencia YOLO sobre el encuadre
        if self.yolo_model is not None:
            try:
                results = self.yolo_model.predict(image, conf=0.20, verbose=False)
                for r in results:
                    for box in r.boxes:
                        xyxy = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cajas_detectadas.append({
                            "bbox": xyxy,
                            "conf": conf,
                            "cls": cls_id
                        })
            except Exception as ex:
                logger.warning(f"Fallo en inferencia YOLO de Farrowing: {ex}")

        # 2. Análisis geométrico y biométrico de postura
        if cajas_detectadas:
            # Seleccionar la caja de mayor área como la cerda madre
            caja_cerda = max(cajas_detectadas, key=lambda b: (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]))
            x1, y1, x2, y2 = caja_cerda["bbox"]
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            ratio = max(bw / bh, bh / bw)
            area_pct = (bw * bh) / (width * height)

            # Clasificación de postura según ratio de elongación lateral y área
            if ratio >= 1.6 or area_pct >= 0.35:
                postura = "decubito_lateral"
            elif ratio >= 1.15:
                postura = "decubito_esternal"
            else:
                postura = "de_pie"

            # Cajas menores representan posibles lechones
            cajas_lechones = [
                b for b in cajas_detectadas
                if b != caja_cerda and ((b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1])) < (bw * bh * 0.25)
            ]
            if cajas_lechones:
                lechones_detectados = max(lechones_detectados, len(cajas_lechones))
        else:
            # Heurística basada en dimensiones nativas si no hay detecciones explícitas
            postura = "decubito_lateral" if (dias_gestacion >= 113 and tiempo_en_postura_min >= 25) else "decubito_esternal"

        # 3. Lógica Zootécnica de Eventos y Probabilidad de Parto
        es_critico = False
        alerta_aplastamiento = False
        alerta_distocia = False
        tipo_evento = "monitoreo_rutinario"
        recomendaciones = []

        if lechones_detectados > 0:
            tipo_evento = "parto_activo"
            probabilidad = round(min(0.99, 0.90 + (lechones_detectados * 0.015)), 2)
            recomendaciones.append(f"Parto activo confirmado con {lechones_detectados} lechón(es) en la jaula.")
            recomendaciones.append("Verificar secado, desinfección de cordón umbilical y calostrado inmediato.")

            # Detección de distocia durante el parto (tiempo excesivo entre lechones)
            if tiempo_en_postura_min >= 55:
                alerta_distocia = True
                es_critico = True
                tipo_evento = "alerta_distocia"
                recomendaciones.append("¡Alerta de distocia! Más de 55 minutos sin nuevo nacimiento. Evaluar palpación o asistencia veterinaria.")

            # Riesgo de aplastamiento si la cerda cambia a postura de pie o esternal brusca
            if postura in ["de_pie", "decubito_esternal"] and lechones_detectados > 0:
                alerta_aplastamiento = True
                es_critico = True
                recomendaciones.append("¡Riesgo de aplastamiento! La cerda está cambiando de posición con lechones sueltos.")

        elif postura == "decubito_lateral" and dias_gestacion >= 112:
            if tiempo_en_postura_min >= 30:
                tipo_evento = "parto_inminente"
                probabilidad = 0.82
                es_critico = (tiempo_en_postura_min >= 60)
                recomendaciones.append("Cerda en decúbito lateral prolongado con fecha probable de parto alcanzada.")
                recomendaciones.append("Mantener encendida la lámpara de calor y alistar kit de atención neonatal.")
            else:
                tipo_evento = "decubito_lateral_inicial"
                probabilidad = 0.55
                recomendaciones.append("Cerda en descanso lateral. Monitorear inicio de contracciones abdominales.")
        elif postura == "de_pie" and dias_gestacion >= 113:
            tipo_evento = "inquietud_preparto"
            probabilidad = 0.45
            recomendaciones.append("Cerda inquieta en fase prodrómica. Comportamiento de anidamiento habitual.")
        else:
            tipo_evento = "monitoreo_rutinario"
            probabilidad = 0.15
            recomendaciones.append("Comportamiento fisiológico normal en jaula de maternidad.")

        return {
            "camara_id": camara_id,
            "postura_cerda": postura,
            "probabilidad_parto": probabilidad,
            "tipo_evento": tipo_evento,
            "es_critico": es_critico,
            "lechones_detectados": lechones_detectados,
            "alerta_aplastamiento": alerta_aplastamiento,
            "alerta_distocia": alerta_distocia,
            "duracion_postura_min": tiempo_en_postura_min,
            "dias_gestacion": dias_gestacion,
            "recomendaciones": recomendaciones,
            "timestamp": datetime.now().isoformat()
        }

    def _respuesta_fallback(self, camara_id: str, error: str) -> Dict[str, Any]:
        """Genera respuesta estándar en caso de fallo en decodificación."""
        return {
            "camara_id": camara_id,
            "postura_cerda": "desconocida",
            "probabilidad_parto": 0.0,
            "tipo_evento": "error_analisis",
            "es_critico": False,
            "lechones_detectados": 0,
            "alerta_aplastamiento": False,
            "alerta_distocia": False,
            "duracion_postura_min": 0,
            "dias_gestacion": 114,
            "recomendaciones": [error],
            "timestamp": datetime.now().isoformat()
        }


# Instancia singleton del analizador
farrowing_analyzer = FarrowingAnalyzer()
