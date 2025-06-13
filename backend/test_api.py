import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_upload():
    response = client.post(
        "/upload",
        files={"file": ("test.bin", b"\x00" * 4096)}
    )
    assert response.status_code == 200
    assert "filename" in response.json()

def test_classify_signal():
    response = client.post(
        "/classify_signal",
        files={"file": ("test.bin", b"\x00" * 4096)}
    )
    assert response.status_code == 200
    assert "signal_type" in response.json()