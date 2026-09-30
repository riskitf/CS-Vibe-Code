"""Guardrail berbasis aturan (bukan LLM) untuk memeriksa input customer
sebelum dikirim ke LLM -- cepat, murah, deterministik."""

MAX_INPUT_LENGTH = 50

# Pola prompt-injection umum, dicek case-sensitive (substring match apa adanya).
PROMPT_INJECTION_PATTERNS = [
    "abaikan intruski",
    "ignore previous",
    "mode developer",
    "cetak system prompt",
    "reveal your prompt",
    "you are now",
]


def check_input(text: str) -> dict:
    """Cek input sebelum dikirim ke LLM. Return {"safe": bool, "reason": str}.
    Kalau tidak aman, safe=False dengan reason yang menjelaskan aturan mana
    yang kena (panjang input atau pola prompt-injection tertentu)."""
    if len(text) > MAX_INPUT_LENGTH:
        return {
            "safe": False,
            "reason": f"Input melebihi {MAX_INPUT_LENGTH} karakter ({len(text)} karakter)",
        }

    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern in text:
            return {
                "safe": False,
                "reason": f"Terdeteksi pola prompt-injection: '{pattern}'",
            }

    return {"safe": True, "reason": ""}
