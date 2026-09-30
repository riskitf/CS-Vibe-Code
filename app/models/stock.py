from pydantic import BaseModel


class StockResponse(BaseModel):
    product_id: str
    name: str
    category: str
    stock: int
    price: int


class ErrorResponse(BaseModel):
    detail: str
