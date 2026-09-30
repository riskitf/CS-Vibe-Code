import csv
from datetime import date, datetime
from pathlib import Path

from app.models.order import OrderResponse

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "orders.csv"

# Catatan CS ditambahkan lewat API disimpan di sini (in-memory), bukan ditulis
# balik ke orders.csv, karena file itu mock data read-only untuk lab.
_notes_override: dict[str, str] = {}


def _load_orders() -> list[dict]:
    """Baca orders.csv. Beberapa baris punya koma yang tidak di-escape di kolom
    `notes` (kolom terakhir), jadi csv.DictReader memecahnya jadi kolom lebih
    ekstra di bawah key `None` (restkey). Gabung balik ke `notes` supaya tidak
    ada key `None` yang bikin OrderResponse(**order) meledak."""
    with open(DATA_PATH, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        extra_columns = row.pop(None, None)
        if extra_columns:
            row["notes"] = ", ".join([row["notes"], *extra_columns])

    return rows


def _find_raw_order(order_id: str) -> dict | None:
    orders = _load_orders()
    normalized = order_id.strip().lower()
    for order in orders:
        if order["order_id"].strip().lower() == normalized:
            return order
    return None


def get_order_by_id(order_id: str) -> OrderResponse | None:
    order = _find_raw_order(order_id)
    if order is None:
        return None
    notes = _notes_override.get(order["order_id"], order["notes"] or None)
    return OrderResponse(**{**order, "notes": notes})


def add_order_note(order_id: str, note: str) -> OrderResponse | None:
    order = _find_raw_order(order_id)
    if order is None:
        return None
    _notes_override[order["order_id"]] = note
    return OrderResponse(**{**order, "notes": note})


def count_orders_by_date(order_date: str) -> list[str]:
    """Cari semua order_id yang order_date-nya sama persis dengan order_date (format YYYY-MM-DD)."""
    orders = _load_orders()
    normalized = order_date.strip()
    return [order["order_id"] for order in orders if order["order_date"].strip() == normalized]


def get_delayed_order_candidates(limit: int | None = None) -> tuple[list[str], list[dict]]:
    """Query order_id dengan status='delayed' dan order_date lebih dari 3 hari dari hari ini.

    Return (order_ids, errors). Baris yang gagal diparse (mis. order_date rusak)
    dicatat sebagai exception di `errors`, tidak menghentikan proses query baris lain.
    """
    orders = _load_orders()
    today = date.today()
    candidate_ids: list[str] = []
    errors: list[dict] = []

    for order in orders:
        order_id = order.get("order_id", "<unknown>")
        try:
            if order["status"].strip().lower() != "delayed":
                continue
            order_date_value = datetime.strptime(order["order_date"].strip(), "%Y-%m-%d").date()
            if (today - order_date_value).days > 3:
                candidate_ids.append(order["order_id"])
        except Exception as e:
            errors.append({"order_id": order_id, "step": "plan_query", "error": str(e)})

    if limit is not None:
        candidate_ids = candidate_ids[:limit]

    return candidate_ids, errors
