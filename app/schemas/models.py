from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    message: str
    chain_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    response: str
    chain_id: str
    input_message: str


class ChainCreateRequest(BaseModel):
    chain_id: str


class MemoryClearRequest(BaseModel):
    chain_id: str
