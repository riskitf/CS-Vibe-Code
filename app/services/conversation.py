import time
from datetime import datetime, timezone
from typing import TypedDict

from app.services import audit


class Message(TypedDict):
    role: str
    content: str
    timestamp: str


SUMMARIZE_PROMPT_INSTRUCTION = (
    "Ringkas percakapan customer service ini jadi maksimal 4 kalimat, sebutkan masalah "
    "customer, order/produk yang dibahas, dan keputusan/info penting yang sudah diberikan. "
    "Kalau nama customer disebutkan di percakapan, sertakan juga namanya di ringkasan. "
    "Jangan tambahkan asumsi baru."
)


class ConversationManager:
    """Menyimpan riwayat chat per session_id di memory (bukan persisten)."""

    MAX_HISTORY_BEFORE_COMPACT = 20
    KEEP_RECENT_ON_COMPACT = 6

    def __init__(self) -> None:
        self._sessions: dict[str, list[Message]] = {}

    def add_message(self, session_id: str, role: str, content: str) -> None:
        history = self._sessions.setdefault(session_id, [])
        if len(history) > self.MAX_HISTORY_BEFORE_COMPACT:
            self.summarize_and_compact(session_id)
            history = self._sessions[session_id]  # summarize_and_compact mengganti list-nya
        history.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def get_context(self, session_id: str, max_turns: int = 12) -> list[Message]:
        """Ambil max_turns pesan terakhir, selalu sertakan pesan pertama di awal
        supaya konteks pembuka (keluhan awal customer) tidak hilang."""
        history = self._sessions.get(session_id, [])
        if not history:
            return []

        first_message = history[0]
        recent_messages = history[-max_turns:]

        if first_message in recent_messages:
            return recent_messages

        return [first_message, *recent_messages]

    def summarize_and_compact(self, session_id: str) -> None:
        """Kirim seluruh riwayat lama (kecuali KEEP_RECENT_ON_COMPACT pesan terakhir)
        ke Gemini untuk diringkas, lalu ganti riwayat lama itu dengan 1 pesan
        system berisi ringkasan. Pesan terakhir tetap dipertahankan apa adanya."""
        history = self._sessions.get(session_id, [])
        keep_recent = self.KEEP_RECENT_ON_COMPACT

        if len(history) <= keep_recent:
            return

        old_messages = history[:-keep_recent]
        recent_messages = history[-keep_recent:]

        if not old_messages:
            return

        start = time.monotonic()
        print(
            f"[ConversationManager] - compaction dimulai utk session={session_id}: "
            f"meringkas {len(old_messages)} pesan lama via Gemini..."
        )

        conversation_text = "\n".join(f"{m['role']}: {m['content']}" for m in old_messages)
        prompt = f"{SUMMARIZE_PROMPT_INSTRUCTION}\n\n{conversation_text}"

        try:
            # Lazy import supaya tidak circular import (agent.py mengimpor conversation.py).
            from app.services.agent import _generate_gemini_text

            summary_text = _generate_gemini_text(prompt).strip()
        except Exception as error:
            print(
                f"[ConversationManager] - compaction GAGAL utk session={session_id} "
                f"setelah {time.monotonic() - start:.2f}s: {error}. Riwayat lama dipertahankan."
            )
            audit.log(
                {
                    "event": "compaction",
                    "session_id": session_id,
                    "status": "failed",
                    "old_message_count": len(old_messages),
                    "error": str(error),
                    "duration_seconds": round(time.monotonic() - start, 3),
                }
            )
            return

        summary_message: Message = {
            "role": "system",
            "content": summary_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._sessions[session_id] = [summary_message, *recent_messages]

        duration = time.monotonic() - start
        print(
            f"[ConversationManager] - compaction selesai utk session={session_id} "
            f"dalam {duration:.2f}s: {len(old_messages)} pesan lama -> 1 pesan "
            f"ringkasan, {len(recent_messages)} pesan terakhir dipertahankan"
        )
        audit.log(
            {
                "event": "compaction",
                "session_id": session_id,
                "status": "success",
                "old_message_count": len(old_messages),
                "kept_recent_count": len(recent_messages),
                "duration_seconds": round(duration, 3),
            }
        )


# Instance bersama (in-memory) dipakai di seluruh app, bukan bikin baru tiap request.
conversation_manager = ConversationManager()
