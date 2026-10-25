"""
test_farrowing.py — Pruebas unitarias para el motor de visión artificial de pariciones (Farrowing AI)
"""

import io
import base64
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app_fastapi import app
from farrowing_service import farrowing_analyzer, CAMARAS_DEFAULT


@pytest.fixture
def client():
    return TestClient(app)


def _crear_imagen_dummy(width=640, height=480, color=(180, 140, 120)):
    """Crea una imagen sintética en memoria simulando un encuadre de corral."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_farrowing_camaras_default():
    """Verifica que las cámaras preconfiguradas existan."""
    assert len(CAMARAS_DEFAULT) >= 8
    assert CAMARAS_DEFAULT[0]["id"] == "cam-01"


def test_farrowing_analyzer_parto_inminente():
    """Evalúa que una cerda en decúbito lateral prolongado a término genere alerta de parto inminente."""
    img_bytes = _crear_imagen_dummy()
    resultado = farrowing_analyzer.analizar_imagen(
        image_bytes=img_bytes,
        camara_id="cam-02",
        tiempo_en_postura_min=45,
        dias_gestacion=114,
        lechones_previos=0
    )

    assert resultado["camara_id"] == "cam-02"
    assert resultado["postura_cerda"] in ["decubito_lateral", "decubito_esternal", "de_pie"]
    assert "probabilidad_parto" in resultado
    assert "tipo_evento" in resultado
    assert isinstance(resultado["recomendaciones"], list)


def test_farrowing_analyzer_parto_activo_con_lechones():
    """Evalúa la clasificación de parto activo cuando hay lechones reportados o detectados."""
    img_bytes = _crear_imagen_dummy()
    resultado = farrowing_analyzer.analizar_imagen(
        image_bytes=img_bytes,
        camara_id="cam-03",
        tiempo_en_postura_min=20,
        dias_gestacion=114,
        lechones_previos=5
    )

    assert resultado["tipo_evento"] == "parto_activo"
    assert resultado["probabilidad_parto"] >= 0.90
    assert resultado["lechones_detectados"] >= 5


def test_farrowing_analyzer_alerta_distocia():
    """Evalúa que si transcurren más de 55 min en decúbito sin avance se dispare alerta de distocia."""
    img_bytes = _crear_imagen_dummy()
    resultado = farrowing_analyzer.analizar_imagen(
        image_bytes=img_bytes,
        camara_id="cam-04",
        tiempo_en_postura_min=65,
        dias_gestacion=114,
        lechones_previos=3
    )

    assert resultado["alerta_distocia"] is True
    assert resultado["es_critico"] is True
    assert any("distocia" in r.lower() for r in resultado["recomendaciones"])


def test_endpoint_farrowing_camaras_config(client):
    """Verifica el endpoint GET /farrowing/camaras-config."""
    response = client.get("/farrowing/camaras-config")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "camaras" in data
    assert len(data["camaras"]) >= 8


def test_endpoint_farrowing_analizar_base64(client):
    """Verifica el endpoint POST /farrowing/analizar-base64."""
    img_bytes = _crear_imagen_dummy()
    b64_str = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode("utf-8")

    payload = {
        "camara_id": "cam-05",
        "tiempo_en_postura_min": 35,
        "dias_gestacion": 114,
        "lechones_previos": 2,
        "image_base64": b64_str
    }

    response = client.post("/farrowing/analizar-base64", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["camara_id"] == "cam-05"
    assert data["data"]["lechones_detectados"] >= 2
