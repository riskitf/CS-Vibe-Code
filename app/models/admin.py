from pydantic import BaseModel


class FollowupDraft(BaseModel):
    order_id: str
    customer_name: str | None = None
    email: str | None = None
    draft_message: str
    provider: str


class FollowupError(BaseModel):
    order_id: str
    step: str
    error: str


class RunFollowupCheckResponse(BaseModel):
    drafts: list[FollowupDraft]
    errors: list[FollowupError]


class MultiAgentDraft(BaseModel):
    order_id: str
    draft_message: str


class RunFollowupCheckMultiAgentResponse(BaseModel):
    drafts: list[MultiAgentDraft]
    errors: list[FollowupError]


class ChunkPreview(BaseModel):
    chunk_id: str
    preview: str
    length: int


class ChunksPreviewResponse(BaseModel):
    chunk_count: int
    chunks: list[ChunkPreview]


class SearchResult(BaseModel):
    text: str
    source: str | None = None
    score: float


class SearchPreviewResponse(BaseModel):
    query: str
    department_filter: str | None = None
    results: list[SearchResult]


class GuardrailTestResponse(BaseModel):
    safe: bool
    reason: str


class PiiPreviewRequest(BaseModel):
    text: str


class PiiPreviewResponse(BaseModel):
    text_masked: str
    mapping: dict[str, str]


class AuditLogResponse(BaseModel):
    count: int
    entries: list[dict]


class AttackTargetRequest(BaseModel):
    message: str


class AttackTargetResponse(BaseModel):
    reply: str
