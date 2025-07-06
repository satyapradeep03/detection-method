import pytest
from fastapi.testclient import TestClient
from main import app, WEIGHTS
import io
import numpy as np
from PIL import Image
import requests

client = TestClient(app)

def create_dummy_image():
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf

def test_analyze_file():
    img = create_dummy_image()
    response = client.post("/analyze/file", files={"file": ("test.png", img, "image/png")})
    assert response.status_code == 200
    data = response.json()
    assert 'prediction' in data
    assert 'metrics' in data
    assert 'visualization' in data

def test_analyze_url(monkeypatch):
    # Mock requests.get to return a dummy image
    class DummyResp:
        status_code = 200
        def iter_content(self, chunk_size):
            img = create_dummy_image().getvalue()
            yield img
    monkeypatch.setattr(requests, 'get', lambda *a, **k: DummyResp())
    response = client.post("/analyze/url", json={"url": "http://fake-url.com/test.png"})
    assert response.status_code == 200
    data = response.json()
    assert 'prediction' in data
    assert 'metrics' in data
    assert 'visualization' in data

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert 'total_requests' in data
    assert 'system' in data

def test_config_get_set():
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert 'facexray' in data
    new_weights = {k: 0.25 for k in WEIGHTS}
    response = client.post("/config", json=new_weights)
    assert response.status_code == 200
    data = response.json()
    assert 'facexray' in data

def test_analyze_file_bad_file():
    response = client.post("/analyze/file", files={"file": ("bad.txt", io.BytesIO(b"notanimage"), "text/plain")})
    assert response.status_code in (400, 500) 