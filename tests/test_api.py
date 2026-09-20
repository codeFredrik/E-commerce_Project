from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_classify_endpoint():
    payload = {
        "observations": [
            {"product_id": 101, "order_date": "2024-01-01", "price": 50.0},
            {"product_id": 101, "order_date": "2024-01-02", "price": 50.0},
            {"product_id": 101, "order_date": "2024-01-03", "price": 70.0},
            {"product_id": 101, "order_date": "2024-01-04", "price": 75.0},
            {"product_id": 101, "order_date": "2024-01-05", "price": 50.0}
        ]
    }
    response = client.post("/api/v1/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["product_id"] == 101
    assert len(data[0]["explanations"]) == 5

def test_benchmark_endpoint():
    response = client.get("/api/v1/benchmark")
    assert response.status_code == 200
    data = response.json()
    assert "metrics_table" in data
    assert "dataset_summary" in data
