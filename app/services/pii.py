"""Masking PII: regex (email/HP/NIK/kartu, cepat & deterministik) + NER
spaCy (nama orang, lebih fleksibel tapi tidak sempurna).

Model NER (id_core_news_sm) BUKAN bawaan spaCy -- diinstal lewat
requirements-hari3.txt (wheel dari Hugging Face). Butuh Python 3.11/3.12 dan
spacy>=3.7,<3.8 (lihat venv-py312/ di root project). Kalau model itu tidak
tersedia di environment yang menjalankan modul ini (mis. Python 3.14 di
environment utama project), masking nama otomatis dilewati (fallback ke
regex saja) -- lihat _get_nlp().

Catatan format placeholder: dinormalisasi konsisten jadi "[tipe N]" untuk
semua kategori (mis. [email1], [telp1], [nik1], [kartu1], [nama1]), bukan
campuran "[email 1]"/"telp1" tanpa kurung seperti draf awal, supaya
predictable dipakai downstream (logging, testing, dsb).
"""

import re

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Nomor HP Indonesia: +62/62/0 diikuti 8 lalu 7-11 digit lagi.
PHONE_PATTERN = re.compile(r"(?:\+62|62|0)8\d{7,11}")

# Kandidat NIK/kartu kredit: 13-19 digit, boleh diselingi spasi/strip di antar digit.
NUMBER_SEQUENCE_PATTERN = re.compile(r"(?<!\d)\d(?:[ -]?\d){11,18}(?!\d)")

NER_MODEL_NAME = "id_core_news_sm"

_nlp_model = None
_nlp_load_attempted = False


def _luhn_check(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _get_nlp():
    """Lazy-load model NER id_core_news_sm sekali saja (cache modul-level).
    Return None kalau spaCy/model-nya tidak tersedia di environment ini,
    supaya mask_pii() tetap jalan dengan fallback regex-only, bukan crash."""
    global _nlp_model, _nlp_load_attempted

    if _nlp_load_attempted:
        return _nlp_model

    _nlp_load_attempted = True
    try:
        import spacy

        _nlp_model = spacy.load(NER_MODEL_NAME)
    except Exception as error:
        print(
            f"[pii] - model NER '{NER_MODEL_NAME}' tidak tersedia ({error}), "
            "masking nama (NER) dilewati, fallback ke regex saja. Jalankan lewat "
            "venv-py312 (lihat requirements-hari3.txt) untuk mengaktifkan deteksi nama."
        )
        _nlp_model = None

    return _nlp_model


def mask_pii(text: str) -> tuple[str, dict]:
    """Sensor email, nomor HP Indonesia, NIK, nomor kartu kredit (regex), lalu
    nama orang (NER spaCy) di `text`. Regex jalan DULU, baru NER -- supaya
    angka/email yang kebetulan nempel di sekitar nama tidak mengacaukan span
    NER, dan supaya NER tidak perlu tahu-menahu soal pola PII yang lain.

    Kartu kredit vs NIK dibedakan lewat Luhn check: sequence 13-19 digit yang
    lolos Luhn -> kartu kredit. Sisa sequence 16 digit yang TIDAK lolos Luhn
    -> NIK. Sequence non-16-digit yang tidak lolos Luhn bukan kartu maupun
    NIK yang valid, jadi tidak disensor.

    NER (label PERSON) TIDAK sempurna (F-score model ini sekitar 0.69) --
    kadang nama kelewat (tidak terdeteksi), kadang kata biasa salah terdeteksi
    sebagai nama. Jangan dianggap jaminan mutlak semua nama tersensor.

    Return (text_tersensor, mapping) -- mapping: {placeholder: nilai_asli}.
    Nilai yang sama muncul berulang kali akan pakai placeholder yang sama.
    """
    mapping: dict[str, str] = {}
    reverse_lookup: dict[tuple[str, str], str] = {}  # (category, original) -> placeholder
    counters = {"email": 0, "telp": 0, "nik": 0, "kartu": 0, "nama": 0}

    def _get_placeholder(category: str, original: str) -> str:
        key = (category, original)
        if key in reverse_lookup:
            return reverse_lookup[key]
        counters[category] += 1
        placeholder = f"[{category}{counters[category]}]"
        reverse_lookup[key] = placeholder
        mapping[placeholder] = original
        return placeholder

    result = text

    # --- Step 1-4: regex (email, HP, NIK/kartu via Luhn) ---
    def _replace_email(match: re.Match) -> str:
        return _get_placeholder("email", match.group(0))

    result = EMAIL_PATTERN.sub(_replace_email, result)

    def _replace_phone(match: re.Match) -> str:
        return _get_placeholder("telp", match.group(0))

    result = PHONE_PATTERN.sub(_replace_phone, result)

    def _replace_number_sequence(match: re.Match) -> str:
        original = match.group(0)
        digits_only = re.sub(r"[ -]", "", original)

        if _luhn_check(digits_only):
            return _get_placeholder("kartu", original)

        if len(digits_only) == 16:
            return _get_placeholder("nik", original)

        return original

    result = NUMBER_SEQUENCE_PATTERN.sub(_replace_number_sequence, result)

    # --- Step 5: NER (nama orang) -- jalan setelah regex ---
    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(result)
        person_entities = sorted(
            (ent for ent in doc.ents if ent.label_ == "PERSON"),
            key=lambda ent: ent.start_char,
        )

        if person_entities:
            pieces = []
            cursor = 0
            for ent in person_entities:
                pieces.append(result[cursor : ent.start_char])
                pieces.append(_get_placeholder("nama", ent.text))
                cursor = ent.end_char
            pieces.append(result[cursor:])
            result = "".join(pieces)

    return result, mapping


if __name__ == "__main__":
    sample = (
        "Saya budi, email budi@mail.com, hp 081234567890, refund ke kartu "
        "4111 1111 1111 1111, nik 3603600987980009"
    )
    masked_text, mapping = mask_pii(sample)

    print("Teks asli:")
    print(sample)
    print()
    print("Teks tersensor:")
    print(masked_text)
    print()
    print("Mapping:")
    for placeholder, original in mapping.items():
        print(f"  {placeholder} -> {original}")

    print()
    print("=" * 70)
    print("DEMO KETERBATASAN NER (buat ditunjukkan ke kelas)")
    print("=" * 70)
    demo_sentences = [
        "Nama saya Budi Santoso, tinggal di Jakarta.",
        "Saya Siti Amalia ingin komplain soal pesanan.",
        "Halo, saya Ahmad Fauzi, order saya ORD1004.",
        "Silakan hubungi Rina Wijaya di kantor pusat.",
    ]
    for sentence in demo_sentences:
        masked, m = mask_pii(sentence)
        print(f"asli   : {sentence}")
        print(f"tersensor: {masked}")
        print(f"mapping: {m}")
        print()
    print(
        "Perhatikan: 'Budi Santoso' & 'Ahmad Fauzi' bisa KELEWAT tidak tersensor "
        "(model melabelinya bukan PERSON), sementara kata biasa seperti 'Halo' / "
        "'Silakan' kadang malah SALAH terdeteksi sebagai nama orang. Ini konsisten "
        "dengan F-score model id_core_news_sm yang cuma sekitar 0.69 -- regex "
        "(email/HP/NIK/kartu) jauh lebih presisi karena polanya jelas, sedangkan "
        "NER buat nama orang inherently lebih ambigu."
    )
