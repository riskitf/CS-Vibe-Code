from fastapi import APIRouter, HTTPException

from app.models.stock import StockResponse
from app.services import stock_service

router = APIRouter(prefix="/stock", tags=["stock"])


@router.get("/{product_name}", response_model=StockResponse)
def get_stock(product_name: str) -> StockResponse:
    stock = stock_service.get_stock_by_name(product_name)
    if stock is None:
        raise HTTPException(status_code=404, detail=f"Produk '{product_name}' tidak ditemukan")
    return stock
