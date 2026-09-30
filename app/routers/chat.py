from fastapi import APIRouter, HTTPException

from app.models.chat import ChatRequest, ChatResponse
from app.services.agent import AgentUnavailableError, call_agent

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        reply = call_agent(payload.session_id, payload.message)
    except AgentUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Asisten sedang tidak bisa diakses (Gemini & Anthropic gagal), coba lagi nanti",
        )
    return ChatResponse(session_id=payload.session_id, reply=reply)
