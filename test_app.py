import os

os.environ["SKIP_VOICE_LOAD"] = "1"

from fastapi.testclient import TestClient
from tts_service.serve import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["skip_voice_load"] is True
