import json
from pathlib import Path

from app.models.stock import StockResponse

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "products.json"


def _load_products() -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_stock_by_name(product_name: str) -> StockResponse | None:
    """Cari produk berdasarkan nama (case-insensitive). Exact match dulu;
    kalau tidak ketemu, fallback ke substring match (mis. 'Keyboard K2' tetap
    match 'Mechanical Keyboard K2') supaya tool-calling LLM yang menyingkat
    nama produk tidak selalu gagal dan berisiko mendorong model berhalusinasi
    saat tool gagal."""
    products = _load_products()
    normalized = product_name.strip().lower()

    for product in products:
        if product["name"].strip().lower() == normalized:
            return StockResponse(**product)

    for product in products:
        if normalized in product["name"].strip().lower():
            return StockResponse(**product)

    return None
