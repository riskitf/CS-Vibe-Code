from fastapi import APIRouter, HTTPException, Path
from typing_extensions import Annotated

from app.models.order import AddOrderNoteRequest, OrderResponse
from app.services import order_service

router = APIRouter(prefix="/orders", tags=["orders"])

OrderIdPath = Annotated[
    str,
    Path(pattern=r"^ORD\d+$", description="Order ID dengan format ORD diikuti angka, misal ORD1001"),
]


@router.get("/{order_id}", response_model=OrderResponse)
def get_order(order_id: OrderIdPath) -> OrderResponse:
    order = order_service.get_order_by_id(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"Order '{order_id}' tidak ditemukan")
    return order


@router.post("/{order_id}/note", response_model=OrderResponse)
def add_order_note(order_id: OrderIdPath, payload: AddOrderNoteRequest) -> OrderResponse:
    order = order_service.add_order_note(order_id, payload.note)
    if order is None:
        raise HTTPException(status_code=404, detail=f"Order '{order_id}' tidak ditemukan")
    return order
