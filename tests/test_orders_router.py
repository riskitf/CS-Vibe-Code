from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_get_order_found_returns_200_with_order_detail():
    response = client.get("/orders/ORD1001")

    assert response.status_code == 200
    body = response.json()
    assert body["order_id"] == "ORD1001"
    assert body["customer_name"] == "Siti Amalia"
    assert body["product_id"] == "P002"
    assert body["status"] == "delivered"


def test_get_order_not_found_returns_404():
    response = client.get("/orders/ORD9999")

    assert response.status_code == 404
    assert "tidak ditemukan" in response.json()["detail"]


def test_get_order_invalid_format_returns_422():
    response = client.get("/orders/abc123")

    assert response.status_code == 422
