"""
test_ai_pipeline.py — Suite de pruebas automatizadas para el microservicio de IA (Porci-Integral).

Valida:
1. Telemetría de hardware y estado de salud en /salud.
2. Catálogo de clases zootécnicas y umbral de confianza en /clases.
3. Diagnóstico de balanceo del dataset en /dataset/diagnostico.
4. Información técnica del modelo en /modelos/info.
5. Inferencia MobileNetV2 con TTA vectorizado y medición de latencia en /reconocer_base64.
6. Pipeline biométrico de 2 fases (YOLOv8 + MobileNetV2) en /reconocer-completo.
7. Autenticación y seguridad de API Key.
"""

import io
import base64
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app_fastapi import app, API_KEY


client = TestClient(app)


def _crear_imagen_sintetica_base64(color=(220, 180, 170), size=(224, 224)) -> str:
    """
    Genera una imagen sintética en memoria y la codifica en base64 para pruebas de inferencia.

    @param color Tupla RGB que simula el tono de piel del ganado porcino (defecto rosáceo).
    @param size Dimensiones de la imagen (ancho, alto).

    @returns {str} Cadena base64 con prefijo data:image/jpeg;base64.

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def test_salud_telemetria_hardware():
    """
    Verifica que el endpoint /salud retorne telemetría completa del sistema anfitrión (CPU y RAM).

    @returns {None}

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    response = client.get("/salud")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "telemetria" in data
    telemetry = data["telemetria"]
    assert "cpu_count_logical" in telemetry
    assert telemetry["cpu_count_logical"] > 0
    assert "ram_total_mb" in telemetry
    assert telemetry["ram_total_mb"] > 0
    assert "ram_used_pct" in telemetry
    assert data["clases_disponibles"] > 0
    assert data["threshold"] > 0


def test_clases_disponibles():
    """
    Verifica que el endpoint /clases retorne la lista de cerdas registradas en classes.json.

    @returns {None}

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    response = client.get("/clases")
    assert response.status_code == 200
    data = response.json()
    assert "clases" in data
    assert isinstance(data["clases"], list)
    assert len(data["clases"]) > 0
    assert data["total"] == len(data["clases"])


def test_dataset_diagnostico_zootecnico():
    """
    Verifica que el endpoint /dataset/diagnostico genere métricas de balanceo fotográfico.

    @returns {None}

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    response = client.get("/dataset/diagnostico")
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    if data["success"]:
        assert "total_imagenes" in data
        assert "total_clases" in data
        assert "recomendaciones_zootecnicas" in data
        assert isinstance(data["recomendaciones_zootecnicas"], list)


def test_modelos_info_tecnica():
    """
    Verifica que el endpoint /modelos/info reporte metadatos de arquitectura y versiones.

    @returns {None}

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    response = client.get("/modelos/info")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "modelo_clasificador" in data
    clasificador = data["modelo_clasificador"]
    assert clasificador["dimensiones_entrada"] == [224, 224]
    assert clasificador["total_clases"] > 0
    assert "detector_yolo" in data
    assert "historial_registro" in data


def test_inferencia_reconocer_base64_con_latencia():
    """
    Verifica la inferencia biométrica en /reconocer_base64 evaluando Top-3 y métrica de latencia en ms.

    @returns {None}

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    img_b64 = _crear_imagen_sintetica_base64()
    response = client.post("/reconocer_base64", json={"imagen": img_b64})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "top1" in data
    assert "confianza_top1" in data
    assert "latencia_ms" in data
    assert data["latencia_ms"] > 0.0
    assert "resultados" in data
    assert len(data["resultados"]) == 3
    # Cada predicción del Top-3 debe contener su latencia y metadatos
    top1 = data["resultados"][0]
    assert "class_name" in top1
    assert "confidence" in top1
    assert "latency_ms" in top1
    assert top1["tta_applied"] is True


def test_seguridad_api_key_reconocer_completo():
    """
    Valida que el endpoint /reconocer-completo rechace peticiones sin autenticación adecuada.

    @returns {None}

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    img_b64 = _crear_imagen_sintetica_base64()
    # Si API_KEY está configurada, una petición sin cabecera debe responder 401
    if API_KEY:
        response = client.post("/reconocer-completo", json={"imagen": img_b64})
        assert response.status_code == 401
        assert "Acceso no autorizado" in response.json()["detail"]


def test_pipeline_completo_dos_fases_yolo_mobilenet():
    """
    Valida el pipeline biométrico híbrido (YOLOv8 + MobileNetV2) con desglose de latencias parciales.

    @returns {None}

    @author Jhon Javer Lozano Arce

    @date 25 de septiembre de 2026
    """
    img_b64 = _crear_imagen_sintetica_base64()
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    payload = {
        "imagen": img_b64,
        "incluir_deteccion": True,
        "umbral": 0.50
    }
    response = client.post("/reconocer-completo", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "identified_as" in data
    assert "tensorflow_predictions" in data
    assert "yolo_detections" in data
    assert "latencia_total_ms" in data
    assert data["latencia_total_ms"] > 0.0
    assert "latencia_mobilenet_ms" in data
    assert "latencia_yolo_ms" in data
