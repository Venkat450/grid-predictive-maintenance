import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from api.app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_endpoint():
    payload = {
        "air_temperature": 300,
        "process_temperature": 310,
        "rotational_speed": 1500,
        "torque": 40,
        "tool_wear": 120,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "failure_probability" in body
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
