import time
import uuid

from fastapi import APIRouter, HTTPException, Query

from app.models.admin import (
    AttackTargetRequest,
    AttackTargetResponse,
    AuditLogResponse,
    ChunkPreview,
    ChunksPreviewResponse,
    GuardrailTestResponse,
    PiiPreviewRequest,
    PiiPreviewResponse,
    RunFollowupCheckMultiAgentResponse,
    RunFollowupCheckResponse,
    SearchPreviewResponse,
    SearchResult,
)
from app.services import audit, order_service
from app.services.agent import AgentUnavailableError, call_agent, researcher_agent, run_followup_check, writer_agent
from app.services.guardrails import check_input
from app.services.pii import mask_pii
from app.services.rag import FAQ_PDF_PATH, chunk_document, search_similar

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/run-followup-check", response_model=RunFollowupCheckResponse)
def run_followup_check_endpoint(
    limit: int | None = Query(default=None, ge=1, description="Proses cuma N order delayed pertama, untuk testing cepat"),
) -> RunFollowupCheckResponse:
    result = run_followup_check(limit=limit)
    return RunFollowupCheckResponse(drafts=result["drafts"], errors=result["errors"])


@router.post("/run-followup-check-multiagent", response_model=RunFollowupCheckMultiAgentResponse)
def run_followup_check_multiagent_endpoint() -> RunFollowupCheckMultiAgentResponse:
    """Versi multi-agent: query order delayed>3hari (sama seperti run-followup-check),
    lalu researcher_agent ambil & validasi data, writer_agent susun draft."""
    start = time.monotonic()
    print("[multiagent] - mulai: query order delayed & umur > 3 hari...")
    order_ids, errors = order_service.get_delayed_order_candidates()
    print(f"[multiagent] - ditemukan {len(order_ids)} order kandidat, {len(errors)} error saat query")

    print("[multiagent] - memanggil researcher_agent...")
    research = researcher_agent(order_ids)
    errors.extend(research["errors"])

    print("[multiagent] - memanggil writer_agent...")
    drafts = writer_agent(research["results"])

    drafted_ids = {draft["order_id"] for draft in drafts}
    for result in research["results"]:
        if result["order_id"] not in drafted_ids:
            errors.append(
                {
                    "order_id": result["order_id"],
                    "step": "write",
                    "error": "Gagal membuat draft follow-up (Gemini & Anthropic gagal)",
                }
            )

    print(
        f"[multiagent] - selesai dalam {time.monotonic() - start:.2f}s: "
        f"{len(drafts)} draft, {len(errors)} error"
    )
    return RunFollowupCheckMultiAgentResponse(drafts=drafts, errors=errors)


@router.get("/chunks-preview", response_model=ChunksPreviewResponse)
def chunks_preview_endpoint(
    chunk_size: int = Query(default=600, ge=50, description="Ukuran chunk dalam karakter"),
) -> ChunksPreviewResponse:
    chunks = chunk_document(str(FAQ_PDF_PATH), chunk_size=chunk_size)
    previews = [
        ChunkPreview(chunk_id=chunk["chunk_id"], preview=chunk["text"][:200], length=len(chunk["text"]))
        for chunk in chunks
    ]
    return ChunksPreviewResponse(chunk_count=len(previews), chunks=previews)


@router.get("/search-preview", response_model=SearchPreviewResponse)
def search_preview_endpoint(
    q: str = Query(min_length=1, description="Query pencarian"),
    top_k: int = Query(default=3, ge=1, description="Jumlah hasil teratas"),
    department_filter: str | None = Query(
        default=None,
        description="Belum berfungsi penuh (baru aktif di step 7), untuk sekarang biarkan kosong",
    ),
) -> SearchPreviewResponse:
    """Murni retrieval -- tidak memanggil LLM sama sekali, cuma cari chunk
    paling mirip lewat search_similar()."""
    results = search_similar(q, top_k=top_k, department_filter=department_filter)
    return SearchPreviewResponse(
        query=q,
        department_filter=department_filter,
        results=[SearchResult(**r) for r in results],
    )


@router.get("/guardrail-test", response_model=GuardrailTestResponse)
def guardrail_test_endpoint(
    text: str = Query(description="Teks yang mau dicek lolos guardrail check_input() atau tidak"),
) -> GuardrailTestResponse:
    """Demo check_input() lewat Swagger -- murni deteksi berbasis aturan,
    tidak memanggil LLM sama sekali."""
    return GuardrailTestResponse(**check_input(text))


@router.post("/pii-preview", response_model=PiiPreviewResponse)
def pii_preview_endpoint(payload: PiiPreviewRequest) -> PiiPreviewResponse:
    """Demo mask_pii() lewat Swagger -- murni regex, tidak memanggil LLM."""
    text_masked, mapping = mask_pii(payload.text)
    return PiiPreviewResponse(text_masked=text_masked, mapping=mapping)


@router.get("/audit-log", response_model=AuditLogResponse)
def audit_log_endpoint(
    n: int = Query(default=20, ge=1, description="Jumlah baris terakhir dari audit.jsonl"),
) -> AuditLogResponse:
    entries = audit.read_last(n)
    return AuditLogResponse(count=len(entries), entries=entries)


@router.post("/attack-target", response_model=AttackTargetResponse)
def attack_target_endpoint(payload: AttackTargetRequest) -> AttackTargetResponse:
    """Target khusus buat red-teaming otomatis (mis. NVIDIA garak). Session_id
    baru (UUID) dibuat tiap request supaya tiap percobaan serangan berdiri
    sendiri -- tidak ada riwayat yang menumpuk/bocor antar percobaan.

    PERINGATAN: endpoint ini sengaja memanggil call_agent() apa adanya
    (termasuk semua guardrail-nya, supaya efektivitas guardrail ikut teruji),
    dan TIDAK PAKAI AUTENTIKASI -- jangan diekspos ke internet publik, cuma
    untuk red-teaming di lingkungan lab/internal.
    """
    session_id = str(uuid.uuid4())
    try:
        reply = call_agent(session_id, payload.message)
    except AgentUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Asisten sedang tidak bisa diakses (Gemini & Anthropic gagal), coba lagi nanti",
        )
    return AttackTargetResponse(reply=reply)
