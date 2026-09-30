from pydantic import BaseModel, Field


class AddOrderNoteRequest(BaseModel):
    note: str = Field(min_length=1, description="Catatan customer service untuk order ini")


class OrderResponse(BaseModel):
    order_id: str
    customer_name: str
    email: str
    phone: str
    product_id: str
    product_name: str
    status: str
    order_date: str
    notes: str | None = None
