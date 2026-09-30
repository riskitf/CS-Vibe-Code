from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from app.routers import admin, chat, orders, stock  # noqa: E402

app = FastAPI(title="Customer Support Assistant")

app.include_router(stock.router)
app.include_router(orders.router)
app.include_router(chat.router)
app.include_router(admin.router)


@app.get("/health")
def health_check():
    """Endpoint verifikasi awal -- dipakai sebelum Sesi 1 untuk cek setup sudah benar.
    Endpoint asli (stock, orders, chat) di-scaffold di Sesi 1 lewat vibe coding,
    menambah ke file ini, bukan menggantikannya dari nol."""
    return {"status": "ok"}


# Mount paling akhir supaya tidak menutupi route API di atas -- StaticFiles(html=True)
# otomatis serve app/static/index.html untuk GET /.
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
