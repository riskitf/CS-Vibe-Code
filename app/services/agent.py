"""Agent orchestration: Gemini (provider utama untuk fungsi ini) dengan
fallback ke Anthropic/Claude kalau Gemini gagal (exception apa pun, termasuk
rate limit, network error, dsb).

Tool didefinisikan sekali dalam format Anthropic (name/description/input_schema)
sebagai sumber kebenaran, lalu dikonversi manual ke format native Gemini
(function_declarations + parameters, tipe JSON Schema terbatas OBJECT/STRING)
saat dikirim ke Gemini.
"""

import json
import os
import time
from datetime import date, datetime
from pathlib import Path

from app.services import order_service, stock_service
from app.services import audit
from app.services.conversation import Message, conversation_manager
from app.services.guardrails import check_input
from app.services.pii import mask_pii
from app.services.rag import RETRIEVAL_THRESHOLD, search_similar

FOLLOWUP_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "app" / "output" / "followup_drafts.json"
FOLLOWUP_MAX_RETRIES = 3
FOLLOWUP_RETRY_BACKOFF_SECONDS = 2
FOLLOWUP_LLM_TIMEOUT_SECONDS = 30

GUARDRAIL_REJECTION_MESSAGE = "Maaf, permintaan Anda tidak dapat diproses"
SECRET_BLOCKED_MESSAGE = "Maaf, permintaan Anda tidak dapat diproses"

KB_NOT_FOUND_MESSAGE = "Maaf, saya tidak menemukan informasi itu di dokumen kami"

# Verifikasi phone_last4 buat lookup_order: maks 3 percobaan salah per session_id,
# ditegakkan oleh kode di sini (bukan oleh LLM).
MAX_PHONE_VERIFICATION_ATTEMPTS = 3
VERIFICATION_BLOCKED_MESSAGE = (
    "Maaf, Anda sudah melewati batas percobaan verifikasi untuk sesi ini. "
    "Silakan hubungi Customer Support kami secara langsung untuk bantuan lebih lanjut."
)
_phone_verification_attempts: dict[str, int] = {}

# Kata kunci rahasia yang tidak boleh bocor ke jawaban (mis. kode diskon internal).
SECRET_KEYWORDS = ["staff-disc"]
# Ukuran & step window buat cek potongan SYSTEM_PROMPT bocor verbatim ke jawaban.
_SYSTEM_PROMPT_LEAK_CHUNK_SIZE = 40
_SYSTEM_PROMPT_LEAK_STEP = 20

SYSTEM_PROMPT = (
    "Kamu adalah asisten customer support toko online. Jawab dalam bahasa Indonesia, "
    "singkat dan jelas. Gunakan tool yang tersedia sebelum menjawab, jangan menebak: "
    "check_stock/check_order_day/lookup_order untuk data spesifik (stok produk, status/detail "
    "SATU order, jumlah order per tanggal), dan search_knowledge_base untuk pertanyaan "
    "kebijakan & prosedur umum toko (garansi, refund, pengiriman, metode pembayaran, cara "
    "retur, dll).\n\n"
    "Cakupanmu HANYA seputar stok produk, status/detail order, dan kebijakan/prosedur toko "
    "ini (lewat search_knowledge_base). Untuk apa pun di luar itu — termasuk basa-basi/sapaan "
    "seperti 'halo', 'apa kabar', obrolan ringan, pengetahuan umum, berita, atau topik lain "
    "apa pun — JANGAN ladeni isinya sama sekali dan JANGAN balas basa-basi itu (misal jangan "
    "jawab 'kabar baik'). Cukup balas dengan satu kalimat penolakan sopan yang mengarahkan ke "
    "topik stok/order/kebijakan toko, tanpa memanggil tool apa pun, setiap kali pesan user "
    "tidak secara eksplisit meminta itu.\n\n"
    "Pengecualian: kalau customer bertanya soal identitasnya sendiri (mis. 'saya siapa?', "
    "'nama saya siapa?') dan nama/info itu memang pernah dia sebutkan sendiri sebelumnya di "
    "riwayat percakapan ini, jawab pakai info itu -- ini masih bagian dari layanan customer, "
    "bukan topik di luar cakupan. Kalau namanya belum pernah disebutkan di riwayat, bilang "
    "belum tahu namanya, jangan mengarang.\n\n"
    "PENTING soal sitasi: kalau hasil tool search_knowledge_base memuat teks '(sumber: ...)', "
    "itu adalah sitasi sumber dokumen -- PERTAHANKAN sitasi itu APA ADANYA (jangan dihapus, "
    "diringkas, atau diparafrase) saat menyusun jawaban akhir untuk customer."
)


def _mask_dict_pii(result: dict) -> tuple[dict, bool]:
    """Sensor PII di semua field string sebuah dict hasil tool (mis. lookup_order,
    search_knowledge_base). Return (dict_baru, ada_yang_disensor)."""
    masked_result = dict(result)
    masked_any = False

    for key, value in result.items():
        if isinstance(value, str):
            masked_value, mapping = mask_pii(value)
            if mapping:
                masked_any = True
            masked_result[key] = masked_value

    return masked_result, masked_any


def _contains_secret(text: str) -> bool:
    """Deteksi kebocoran rahasia di jawaban akhir: kata kunci rahasia (mis.
    kode diskon internal) atau potongan verbatim dari SYSTEM_PROMPT."""
    lowered = text.lower()
    for keyword in SECRET_KEYWORDS:
        if keyword.lower() in lowered:
            return True

    for start in range(0, len(SYSTEM_PROMPT) - _SYSTEM_PROMPT_LEAK_CHUNK_SIZE, _SYSTEM_PROMPT_LEAK_STEP):
        chunk = SYSTEM_PROMPT[start : start + _SYSTEM_PROMPT_LEAK_CHUNK_SIZE]
        if chunk.strip() and chunk in text:
            return True

    return False


# Format Anthropic: name, description, input_schema (sumber kebenaran untuk semua provider).
ANTHROPIC_TOOLS = [
    {
        "name": "check_stock",
        "description": "Cek stok dan harga sebuah produk berdasarkan nama produk.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "Nama produk yang ingin dicek stoknya, misal 'Wireless Mouse X1'.",
                }
            },
            "required": ["product_name"],
        },
    },
    {
        "name": "lookup_order",
        "description": (
            "Cek detail dan status sebuah order berdasarkan order_id. Tanpa phone_last4 yang "
            "cocok, tool ini HANYA mengembalikan order_id/product_name/status (data sensitif "
            "seperti nama, email, nomor HP, catatan CS disembunyikan). SEBELUM menampilkan "
            "detail lengkap ke customer, kamu WAJIB minta customer menyebutkan HANYA 4 digit "
            "TERAKHIR nomor HP yang terdaftar di order tsb (bukan nomor lengkap -- nomor HP "
            "lengkap yang customer ketik akan otomatis tersensor jadi placeholder seperti "
            "'[telp1]' sebelum sampai ke kamu, jadi memang tidak bisa dipakai untuk verifikasi "
            "ini). Verifikasi dilakukan oleh kode tool ini sendiri, bukan olehmu -- cukup "
            "teruskan 4 digit yang disebutkan customer lewat parameter phone_last4."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "ID order yang ingin dicek, format ORD diikuti angka, misal 'ORD1001'.",
                },
                "phone_last4": {
                    "type": "string",
                    "description": (
                        "4 digit TERAKHIR nomor HP yang terdaftar di order tsb, disebutkan "
                        "sendiri oleh customer buat verifikasi. Kosongkan/jangan isi kalau "
                        "customer belum menyebutkannya -- jangan menebak atau mengarang nilainya."
                    ),
                },
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "check_order_day",
        "description": "Hitung berapa banyak order yang dibuat pada satu tanggal tertentu.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_date": {
                    "type": "string",
                    "description": "Tanggal order yang ingin dicek, format YYYY-MM-DD, misal '2026-09-10'.",
                }
            },
            "required": ["order_date"],
        },
    },
    {
        "name": "search_knowledge_base",
        "description": (
            "Cari jawaban dari dokumen KEBIJAKAN & PROSEDUR toko (kebijakan pengiriman, "
            "kebijakan pengembalian/refund, garansi produk, metode pembayaran, cara melacak "
            "pesanan secara umum, FAQ produk). PENTING -- beda dengan lookup_order (status/"
            "detail SATU order spesifik milik customer) dan check_stock (stok SATU produk "
            "spesifik): pakai search_knowledge_base untuk pertanyaan kebijakan/prosedur umum "
            "(mis. 'berapa lama garansi headset', 'apa bisa COD', 'cara retur barang'), BUKAN "
            "untuk menanyakan status order tertentu atau stok produk tertentu."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Pertanyaan customer tentang kebijakan/prosedur yang ingin dicari jawabannya.",
                }
            },
            "required": ["question"],
        },
    },
]


def _anthropic_schema_to_gemini_schema(schema: dict) -> dict:
    """Konversi JSON Schema sederhana (yang dipakai di input_schema Anthropic)
    ke format Schema native Gemini (tipe huruf besar: OBJECT, STRING, dst)."""
    type_map = {
        "object": "OBJECT",
        "string": "STRING",
        "number": "NUMBER",
        "integer": "INTEGER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
    }
    gemini_schema: dict = {"type": type_map.get(schema.get("type", "object"), "OBJECT")}

    if "properties" in schema:
        gemini_schema["properties"] = {
            key: _anthropic_schema_to_gemini_schema(value) for key, value in schema["properties"].items()
        }
    if "description" in schema:
        gemini_schema["description"] = schema["description"]
    if "required" in schema:
        gemini_schema["required"] = schema["required"]
    if "items" in schema:
        gemini_schema["items"] = _anthropic_schema_to_gemini_schema(schema["items"])

    return gemini_schema


def _build_gemini_tools() -> list:
    from google.generativeai.types import FunctionDeclaration, Tool

    declarations = [
        FunctionDeclaration(
            name=tool["name"],
            description=tool["description"],
            parameters=_anthropic_schema_to_gemini_schema(tool["input_schema"]),
        )
        for tool in ANTHROPIC_TOOLS
    ]
    return [Tool(function_declarations=declarations)]


def _run_tool(
    tool_name: str,
    tool_input: dict,
    tools_called: list[str] | None = None,
    events: list[str] | None = None,
    session_id: str | None = None,
) -> dict:
    """Jalankan tool yang sesuai dan return hasilnya sebagai dict JSON-serializable.
    Kalau `tools_called`/`events` diberikan (dari call_agent), dicatat ke situ
    untuk keperluan audit. `session_id` dipakai khusus lookup_order untuk
    verifikasi phone_last4 (dilewati/dianggap trusted kalau None, mis. dipanggil
    dari job internal seperti run_followup_check/researcher_agent)."""
    print(f"[tool:{tool_name}] - menjalankan dengan input {tool_input}...")

    if tools_called is not None:
        tools_called.append(tool_name)

    if tool_name == "check_stock":
        product_name = tool_input.get("product_name", "")
        stock = stock_service.get_stock_by_name(product_name)
        if stock is None:
            return {"found": False, "error": f"Produk '{product_name}' tidak ditemukan"}
        return {"found": True, **stock.model_dump()}

    if tool_name == "lookup_order":
        order_id = tool_input.get("order_id", "")
        phone_last4 = (tool_input.get("phone_last4") or "").strip()

        order = order_service.get_order_by_id(order_id)
        if order is None:
            return {"found": False, "error": f"Order '{order_id}' tidak ditemukan"}

        # Dipanggil dari job internal (tidak ada session_id) -> trusted, skip verifikasi.
        if session_id is None:
            result, masked_any = _mask_dict_pii({"found": True, **order.model_dump()})
            if masked_any:
                print("[guardrail] - tool_pi_masked (lookup_order)")
                if events is not None:
                    events.append("tool_pi_masked")
            return result

        attempts_so_far = _phone_verification_attempts.get(session_id, 0)
        if attempts_so_far >= MAX_PHONE_VERIFICATION_ATTEMPTS:
            print(f"[guardrail] - lookup_order_verification_blocked (session={session_id})")
            if events is not None:
                events.append("lookup_order_verification_blocked")
            audit.log(
                {
                    "event": "lookup_order_verification_blocked",
                    "session_id": session_id,
                    "order_id": order_id,
                    "attempts": attempts_so_far,
                }
            )
            return {"found": False, "error": VERIFICATION_BLOCKED_MESSAGE}

        real_last4 = (order.phone or "")[-4:]
        is_match = bool(phone_last4) and phone_last4 == real_last4

        if not is_match:
            if phone_last4:
                # Cuma dihitung sebagai "percobaan" kalau customer memang menebak
                # sesuatu -- bukan cuma karena phone_last4 belum diisi sama sekali.
                _phone_verification_attempts[session_id] = attempts_so_far + 1
                print(
                    f"[guardrail] - lookup_order_verification_failed (session={session_id}, "
                    f"percobaan={attempts_so_far + 1}/{MAX_PHONE_VERIFICATION_ATTEMPTS})"
                )
                if events is not None:
                    events.append("lookup_order_verification_failed")
                audit.log(
                    {
                        "event": "lookup_order_verification_failed",
                        "session_id": session_id,
                        "order_id": order_id,
                        "attempts": attempts_so_far + 1,
                    }
                )

            return {
                "found": True,
                "order_id": order.order_id,
                "product_name": order.product_name,
                "status": order.status,
                "butuh_verifikasi": True,
            }

        result, masked_any = _mask_dict_pii({"found": True, "butuh_verifikasi": False, **order.model_dump()})
        if masked_any:
            print("[guardrail] - tool_pi_masked (lookup_order)")
            if events is not None:
                events.append("tool_pi_masked")
        return result

    if tool_name == "check_order_day":
        order_date = tool_input.get("order_date", "")
        order_ids = order_service.count_orders_by_date(order_date)
        return {"found": True, "order_date": order_date, "count": len(order_ids), "order_ids": order_ids}

    if tool_name == "search_knowledge_base":
        question = tool_input.get("question", "")
        result, masked_any = _mask_dict_pii({"found": True, "answer": _search_knowledge_base(question)})
        if masked_any:
            print("[guardrail] - tool_pi_masked (search_knowledge_base)")
            if events is not None:
                events.append("tool_pi_masked")
        return result

    return {"found": False, "error": f"Tool '{tool_name}' tidak dikenal"}


def _history_to_gemini(history: list[Message]) -> list[dict]:
    """Konversi riwayat ConversationManager (role user/assistant/system) ke
    format history Gemini (role user/model). Pesan system digabung sebagai
    catatan teks dari user supaya tidak hilang."""
    gemini_history = []
    for message in history:
        if message["role"] == "assistant":
            role = "model"
        else:
            role = "user"
        content = message["content"]
        if message["role"] == "system":
            content = f"[catatan konteks] {content}"
        gemini_history.append({"role": role, "parts": [content]})
    return gemini_history


def _call_gemini(
    history: list[Message],
    user_message: str,
    tools_called: list[str] | None = None,
    events: list[str] | None = None,
    session_id: str | None = None,
) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name="gemini-flash-latest",
        system_instruction=SYSTEM_PROMPT,
        tools=_build_gemini_tools(),
    )

    chat = model.start_chat(history=_history_to_gemini(history))
    response = chat.send_message(user_message)

    # Loop selama Gemini minta jalankan function/tool.
    while True:
        function_calls = [
            part.function_call
            for candidate in response.candidates
            for part in candidate.content.parts
            if part.function_call
        ]
        if not function_calls:
            break

        function_responses = []
        for function_call in function_calls:
            tool_result = _run_tool(
                function_call.name,
                dict(function_call.args),
                tools_called=tools_called,
                events=events,
                session_id=session_id,
            )
            function_responses.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=function_call.name,
                        response={"result": tool_result},
                    )
                )
            )
        response = chat.send_message(function_responses)

    return response.text


def _history_to_anthropic(history: list[Message]) -> list[dict]:
    """Anthropic hanya kenal role user/assistant; pesan system digabung ke
    pesan user berikutnya sebagai catatan konteks."""
    anthropic_messages = []
    for message in history:
        role = "assistant" if message["role"] == "assistant" else "user"
        content = message["content"]
        if message["role"] == "system":
            content = f"[catatan konteks] {content}"
        anthropic_messages.append({"role": role, "content": content})
    return anthropic_messages


def _call_anthropic(
    history: list[Message],
    user_message: str,
    tools_called: list[str] | None = None,
    events: list[str] | None = None,
    session_id: str | None = None,
) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages = _history_to_anthropic(history)
    messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=ANTHROPIC_TOOLS,
        messages=messages,
    )

    while response.stop_reason == "tool_use":
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_result = _run_tool(
                    block.name,
                    block.input,
                    tools_called=tools_called,
                    events=events,
                    session_id=session_id,
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(tool_result),
                    }
                )
        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=ANTHROPIC_TOOLS,
            messages=messages,
        )

    final_text = "".join(block.text for block in response.content if block.type == "text")
    return final_text


class AgentUnavailableError(Exception):
    """Dilempar kalau Gemini maupun Anthropic gagal (API key hilang, rate
    limit, network error, dsb). Router yang memanggil call_agent harus
    menangkap ini dan mengembalikannya sebagai response error, bukan
    membiarkan exception provider mentah bocor ke client."""


def call_agent(session_id: str, user_message: str) -> str:
    """Proses satu giliran chat: gabungkan riwayat + pesan baru, kirim ke
    Gemini (utama) dengan fallback ke Anthropic kalau Gemini gagal, lalu
    simpan pesan user & jawaban final ke ConversationManager.

    Tiap request dicatat ke audit.jsonl lewat app.services.audit."""
    start = time.monotonic()
    print(f"[call_agent] - session={session_id}: memproses pesan \"{user_message[:80]}\"...")

    input_length = len(user_message)
    events: list[str] = []
    tools_called: list[str] = []

    guardrail_result = check_input(user_message)
    if not guardrail_result["safe"]:
        print(
            f"[call_agent] - session={session_id}: GUARDRAIL menolak input "
            f"({guardrail_result['reason']}) -- tidak dikirim ke LLM, tidak disimpan ke ConversationManager."
        )
        audit.log(
            {
                "session_id": session_id,
                "input_length": input_length,
                "input_masked": None,
                "guardrail": guardrail_result,
                "events": events,
                "tools_called": tools_called,
                "provider": None,
                "duration_seconds": round(time.monotonic() - start, 3),
            }
        )
        return GUARDRAIL_REJECTION_MESSAGE

    # (a) Sensor PII di input user SEBELUM dikirim ke LLM. Versi tersensor ini
    # yang dipakai untuk pemanggilan LLM maupun disimpan ke ConversationManager.
    user_message, input_pii_mapping = mask_pii(user_message)
    if input_pii_mapping:
        print(f"[guardrail] - Input_pii_masked (session={session_id}): {list(input_pii_mapping.keys())}")
        events.append("Input_pii_masked")

    history = conversation_manager.get_context(session_id)

    try:
        print(f"[call_agent] - session={session_id}: memanggil Gemini...")
        final_answer = _call_gemini(
            history, user_message, tools_called=tools_called, events=events, session_id=session_id
        )
        provider_used = "gemini"
    except Exception as gemini_error:
        print(f"[call_agent] - session={session_id}: Gemini gagal ({gemini_error}), fallback ke Anthropic...")
        events.append("provider_switch")
        audit.log(
            {
                "event": "provider_switch",
                "session_id": session_id,
                "from": "gemini",
                "to": "anthropic",
                "reason": str(gemini_error),
            }
        )
        try:
            final_answer = _call_anthropic(
                history, user_message, tools_called=tools_called, events=events, session_id=session_id
            )
            provider_used = "anthropic"
        except Exception as anthropic_error:
            print(f"[call_agent] - session={session_id}: Anthropic juga gagal ({anthropic_error}).")
            audit.log(
                {
                    "session_id": session_id,
                    "input_length": input_length,
                    "input_masked": user_message,
                    "guardrail": guardrail_result,
                    "events": events,
                    "tools_called": tools_called,
                    "provider": None,
                    "duration_seconds": round(time.monotonic() - start, 3),
                    "error": f"Gemini gagal ({gemini_error}) dan Anthropic juga gagal ({anthropic_error})",
                }
            )
            raise AgentUnavailableError(
                f"Gemini gagal ({gemini_error}) dan Anthropic juga gagal ({anthropic_error})"
            ) from anthropic_error

    # (c) Output guardrail: sensor PII dulu di jawaban akhir SEBELUM dikirim ke
    # user, lalu cek kebocoran rahasia (kode internal / potongan SYSTEM_PROMPT).
    final_answer, output_pii_mapping = mask_pii(final_answer)
    if output_pii_mapping:
        print(f"[guardrail] - output_pii_masked (session={session_id}): {list(output_pii_mapping.keys())}")
        events.append("output_pii_masked")

    if _contains_secret(final_answer):
        print(f"[guardrail] - output_secret_blocked (session={session_id}): jawaban diganti total.")
        events.append("output_secret_blocked")
        final_answer = SECRET_BLOCKED_MESSAGE

    conversation_manager.add_message(session_id, "user", user_message)
    conversation_manager.add_message(session_id, "assistant", final_answer)

    duration = time.monotonic() - start
    print(f"[call_agent] - session={session_id}: selesai via {provider_used} dalam {duration:.2f}s")

    audit.log(
        {
            "session_id": session_id,
            "input_length": input_length,
            "input_masked": user_message,
            "guardrail": guardrail_result,
            "events": events,
            "tools_called": tools_called,
            "provider": provider_used,
            "duration_seconds": round(duration, 3),
        }
    )

    return final_answer


def _is_retryable_error(error: Exception) -> bool:
    """Klasifikasi kasar: error rate limit / timeout dianggap retryable,
    error lain (mis. API key hilang, input tidak valid) langsung raise."""
    text = str(error).lower()
    retryable_markers = ("timeout", "timed out", "rate limit", "429", "deadline", "resource_exhausted")
    return any(marker in text for marker in retryable_markers)


def _call_llm_with_retry(label: str, fn, max_retries: int = FOLLOWUP_MAX_RETRIES,
                          backoff_seconds: float = FOLLOWUP_RETRY_BACKOFF_SECONDS):
    """Jalankan fn() dengan retry+backoff kalau errornya rate limit/timeout,
    maks max_retries kali, lalu raise. Log granular sebelum & sesudah tiap percobaan."""
    attempt = 0
    while True:
        attempt += 1
        print(f"{label} (percobaan {attempt}/{max_retries})...")
        attempt_start = time.monotonic()
        try:
            result = fn()
            print(f"{label} -- berhasil dalam {time.monotonic() - attempt_start:.2f}s")
            return result
        except Exception as error:
            elapsed = time.monotonic() - attempt_start
            if not _is_retryable_error(error) or attempt >= max_retries:
                print(f"[FAILED] {label} -- gagal permanen setelah {attempt} percobaan ({elapsed:.2f}s): {error}")
                raise
            print(f"[RETRY] {label} -- gagal ({elapsed:.2f}s): {error} -- retry dalam {backoff_seconds}s")
            time.sleep(backoff_seconds)


def _build_followup_prompt(order: dict) -> str:
    return (
        "Susun draft pesan follow-up yang sopan dan singkat dalam bahasa Indonesia untuk "
        "customer terkait order yang tertunda (delayed). Tulis pesan siap kirim, jangan pakai "
        "placeholder seperti [nama].\n\n"
        f"Nama customer: {order.get('customer_name')}\n"
        f"Order ID: {order.get('order_id')}\n"
        f"Produk: {order.get('product_name')}\n"
        f"Tanggal order: {order.get('order_date')}\n"
        f"Catatan CS sebelumnya: {order.get('notes') or '-'}\n"
    )


def _generate_gemini_text(prompt: str, timeout: int = FOLLOWUP_LLM_TIMEOUT_SECONDS) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(model_name="gemini-flash-latest")
    response = model.generate_content(prompt, request_options={"timeout": timeout})
    return response.text


def _generate_anthropic_text(prompt: str, timeout: int = FOLLOWUP_LLM_TIMEOUT_SECONDS) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=timeout)
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _call_llm_simple(prompt: str) -> str:
    """Wrapper LLM sederhana (satu prompt, tanpa histori/tools) dengan pola
    fallback YANG SAMA PERSIS dengan call_agent(): Gemini dulu, kalau gagal
    fallback ke Anthropic -- reuse _generate_gemini_text/_generate_anthropic_text
    yang sudah ada (timeout 30 detik lewat FOLLOWUP_LLM_TIMEOUT_SECONDS), bukan
    bikin pemanggilan LLM baru."""
    try:
        return _generate_gemini_text(prompt, timeout=FOLLOWUP_LLM_TIMEOUT_SECONDS)
    except Exception as gemini_error:
        print(f"[search_knowledge_base] - Gemini gagal ({gemini_error}), fallback ke Anthropic...")
        return _generate_anthropic_text(prompt, timeout=FOLLOWUP_LLM_TIMEOUT_SECONDS)


def _build_knowledge_base_prompt(question: str, results: list[dict]) -> str:
    numbered_context = "\n\n".join(
        f"[{i}] (sumber: {r['source']})\n{r['text']}" for i, r in enumerate(results, start=1)
    )
    return (
        "Kamu adalah asisten customer support toko online. Jawab pertanyaan customer dalam "
        "bahasa Indonesia, singkat dan jelas, HANYA berdasarkan konteks kutipan dokumen "
        "kebijakan/FAQ di bawah ini.\n\n"
        f"Konteks:\n{numbered_context}\n\n"
        f"Pertanyaan: {question}\n\n"
        "Jawab hanya berdasarkan konteks. Kalau tidak ada, jawab persis kalimat di atas. "
        "Jangan mengarang."
    )


def _search_knowledge_base(question: str) -> str:
    """Alur tool search_knowledge_base: search_similar() -> cek skor top-1
    terhadap RETRIEVAL_THRESHOLD -- kalau tidak lolos, return pesan not-found
    tanpa panggil LLM sama sekali; kalau lolos, susun prompt (top-3 chunk
    bernomor + sumber + pertanyaan + instruksi anti-halusinasi) dan kirim ke
    LLM lewat _call_llm_simple (wrapper sama persis dengan call_agent)."""
    print(f"[search_knowledge_base] - mencari jawaban untuk \"{question[:80]}\"...")
    results = search_similar(question, top_k=3)

    top_score = results[0]["score"] if results else None
    if not results or top_score < RETRIEVAL_THRESHOLD:
        print(
            f"[search_knowledge_base] - skor top-1 ({top_score}) di bawah RETRIEVAL_THRESHOLD "
            f"({RETRIEVAL_THRESHOLD}), tidak memanggil LLM"
        )
        return KB_NOT_FOUND_MESSAGE

    print(
        f"[search_knowledge_base] - skor top-1 ({top_score:.4f}) lolos threshold "
        f"({RETRIEVAL_THRESHOLD}), menyusun prompt ke LLM (timeout {FOLLOWUP_LLM_TIMEOUT_SECONDS}s)..."
    )
    prompt = _build_knowledge_base_prompt(question, results)
    answer = _call_llm_simple(prompt).strip()

    # Sitasi ditambahkan deterministik di sini (bukan diserahkan ke LLM) supaya
    # selalu konsisten formatnya dan tidak pernah hilang/ke-parafrase.
    sources = list(dict.fromkeys(r["source"] for r in results))  # unik, urutan dipertahankan
    citation = f"(sumber: {', '.join(sources)})"

    return f"{answer} {citation}"


def _draft_followup_gemini(order: dict, timeout: int = FOLLOWUP_LLM_TIMEOUT_SECONDS) -> str:
    return _generate_gemini_text(_build_followup_prompt(order), timeout=timeout)


def _draft_followup_anthropic(order: dict, timeout: int = FOLLOWUP_LLM_TIMEOUT_SECONDS) -> str:
    return _generate_anthropic_text(_build_followup_prompt(order), timeout=timeout)


def run_followup_check(limit: int | None = None) -> dict:
    """Loop plan-execute-reflect-revise untuk follow-up order yang delayed > 3 hari:

    - plan: query orders.csv (status=delayed, order_date > 3 hari lalu), potong ke `limit`
      kalau diisi (buat testing cepat)
    - execute: tool lookup_order per order kandidat buat ambil detail lengkap
    - reflect: order yang gagal di-query/lookup dicatat sebagai error, loop tetap lanjut
    - execute: minta LLM (Gemini utama, fallback Anthropic) susun draft follow-up,
      tiap panggilan LLM pakai timeout 30 detik
    - execute: simpan semua draft ke app/output/followup_drafts.json
    - revise: kalau gagal karena rate limit/timeout, retry dgn backoff 2 detik maks 3x,
      raise kalau tetap gagal setelah retry habis
    """
    run_start = time.monotonic()
    print(f"[Plan] Query orders.csv: status=delayed & order_date > 3 hari dari hari ini (limit={limit})...")
    candidate_ids, errors = order_service.get_delayed_order_candidates(limit=limit)
    print(f"[Plan] Selesai: {len(candidate_ids)} order kandidat, {len(errors)} error saat query.")

    drafts: list[dict] = []
    total = len(candidate_ids)

    for i, order_id in enumerate(candidate_ids, start=1):
        order_start = time.monotonic()
        print(f"[{i}/{total}] Memproses order {order_id}...")
        try:
            print(f"[{i}/{total}] Menjalankan tool lookup_order untuk {order_id}...")
            tool_result = _run_tool("lookup_order", {"order_id": order_id})
            if not tool_result.get("found"):
                raise ValueError(tool_result.get("error", f"Order {order_id} tidak ditemukan"))

            try:
                draft_text = _call_llm_with_retry(
                    f"[{i}/{total}] Memanggil LLM (Gemini, timeout {FOLLOWUP_LLM_TIMEOUT_SECONDS}s) "
                    f"untuk draft follow-up {order_id}",
                    lambda tr=tool_result: _draft_followup_gemini(tr),
                )
                provider_used = "gemini"
            except Exception as gemini_error:
                print(f"[{i}/{total}] Gemini gagal total untuk {order_id} ({gemini_error}), fallback ke Anthropic...")
                draft_text = _call_llm_with_retry(
                    f"[{i}/{total}] Memanggil LLM (Anthropic, timeout {FOLLOWUP_LLM_TIMEOUT_SECONDS}s) "
                    f"untuk draft follow-up {order_id}",
                    lambda tr=tool_result: _draft_followup_anthropic(tr),
                )
                provider_used = "anthropic"

            drafts.append(
                {
                    "order_id": order_id,
                    "customer_name": tool_result.get("customer_name"),
                    "email": tool_result.get("email"),
                    "draft_message": draft_text,
                    "provider": provider_used,
                }
            )
            print(f"[{i}/{total}] Selesai order {order_id} dalam {time.monotonic() - order_start:.2f}s")
        except Exception as error:
            print(f"[{i}/{total}] GAGAL order {order_id} setelah {time.monotonic() - order_start:.2f}s: {error}")
            errors.append({"order_id": order_id, "step": "execute", "error": str(error)})
            continue

    print(f"Menyimpan {len(drafts)} draft ke {FOLLOWUP_OUTPUT_PATH}...")
    FOLLOWUP_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FOLLOWUP_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"drafts": drafts, "errors": errors}, f, ensure_ascii=False, indent=2)

    duration = time.monotonic() - run_start
    print(f"[Done] selesai dalam {duration:.2f}s -- {len(drafts)} draft dibuat, {len(errors)} error")

    return {"drafts": drafts, "errors": errors, "duration_seconds": duration}


def researcher_agent(order_ids: list[str]) -> dict:
    """Ambil & validasi data tiap order_id lewat tool lookup_order (+ check_stock
    untuk validasi produknya masih terdaftar), lalu susun jadi dict ringkas.

    Return {"results": [...], "errors": [...]}. Order yang gagal diambil/divalidasi
    (tidak ditemukan, produk tidak dikenal, order_date rusak, dsb) dicatat ke
    `errors`, tidak menghentikan proses order_id lain.
    """
    start = time.monotonic()
    print(f"[researcher_agent] - mulai memproses {len(order_ids)} order: {order_ids}")

    results: list[dict] = []
    errors: list[dict] = []

    for i, order_id in enumerate(order_ids, start=1):
        print(f"[researcher_agent] - [{i}/{len(order_ids)}] mengambil & validasi data {order_id}...")
        try:
            order_result = _run_tool("lookup_order", {"order_id": order_id})
            if not order_result.get("found"):
                raise ValueError(order_result.get("error", f"Order '{order_id}' tidak ditemukan"))

            product_name = order_result.get("product_name")
            stock_result = _run_tool("check_stock", {"product_name": product_name})
            if not stock_result.get("found"):
                raise ValueError(stock_result.get("error", f"Produk '{product_name}' tidak ditemukan"))

            order_date_value = datetime.strptime(order_result["order_date"].strip(), "%Y-%m-%d").date()
            days_delayed = (date.today() - order_date_value).days

            results.append(
                {
                    "order_id": order_id,
                    "customer_name": order_result.get("customer_name"),
                    "product": product_name,
                    "status": order_result.get("status"),
                    "days_delayed": days_delayed,
                }
            )
            print(f"[researcher_agent] - [{i}/{len(order_ids)}] {order_id} valid, delayed {days_delayed} hari")
        except Exception as error:
            print(f"[researcher_agent] - [{i}/{len(order_ids)}] {order_id} GAGAL: {error}")
            errors.append({"order_id": order_id, "step": "research", "error": str(error)})
            continue

    print(
        f"[researcher_agent] - selesai dalam {time.monotonic() - start:.2f}s: "
        f"{len(results)} valid, {len(errors)} error"
    )
    return {"results": results, "errors": errors}


def _build_writer_prompt(order: dict) -> str:
    return (
        "Susun draft pesan follow-up customer service dalam bahasa Indonesia untuk order "
        "berikut. Ketentuan wajib: maksimal 3 kalimat, sopan, sebutkan nomor order-nya, dan "
        "sertakan permintaan maaf atas keterlambatannya. Tulis pesan siap kirim, jangan pakai "
        "placeholder seperti [nama], jangan tambahkan penjelasan lain di luar pesannya.\n\n"
        f"Nama customer: {order.get('customer_name')}\n"
        f"Order ID: {order.get('order_id')}\n"
        f"Produk: {order.get('product')}\n"
        f"Status order: {order.get('status')}\n"
        f"Sudah tertunda: {order.get('days_delayed')} hari\n"
    )


def writer_agent(researcher_output: list[dict]) -> list[dict]:
    """Terima output terstruktur researcher_agent (list of
    {order_id, customer_name, product, status, days_delayed}), lalu minta LLM
    (Gemini utama, fallback Anthropic) menyusun draft follow-up singkat (maks 3
    kalimat) untuk tiap order. Order yang gagal di-draft dilewati, tidak
    menghentikan proses order lain."""
    start = time.monotonic()
    print(f"[writer_agent] - mulai menyusun draft untuk {len(researcher_output)} order...")

    drafts: list[dict] = []

    for i, order in enumerate(researcher_output, start=1):
        order_id = order.get("order_id")
        prompt = _build_writer_prompt(order)
        print(f"[writer_agent] - [{i}/{len(researcher_output)}] memanggil Gemini untuk draft {order_id}...")
        try:
            draft_message = _generate_gemini_text(prompt)
        except Exception as gemini_error:
            print(f"[writer_agent] - [{i}/{len(researcher_output)}] Gemini gagal untuk {order_id} ({gemini_error}), fallback ke Anthropic...")
            try:
                draft_message = _generate_anthropic_text(prompt)
            except Exception as anthropic_error:
                print(f"[writer_agent] - [{i}/{len(researcher_output)}] Anthropic juga gagal untuk {order_id} ({anthropic_error}), dilewati.")
                continue

        drafts.append({"order_id": order_id, "draft_message": draft_message.strip()})
        print(f"[writer_agent] - [{i}/{len(researcher_output)}] draft {order_id} selesai")

    print(f"[writer_agent] - selesai dalam {time.monotonic() - start:.2f}s: {len(drafts)} draft dibuat")
    return drafts
