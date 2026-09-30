"""Chunking dokumen PDF untuk RAG, pakai pdfplumber buat ekstraksi teks.

Catatan ekstraksi: `page.extract_text()` mengekstrak teks per baris tanpa
memahami struktur tabel. Untuk tabel Garansi Produk di faq-produk.pdf, ini
bikin kolom-kolom nempel tanpa spasi/delimiter yang jelas, mis.
"Elektronik (webcam, power bank, monitor)12 bulan Cacat produksi & kerusakan
komponen" -- nama kategori, angka masa garansi, dan cakupan garansi jadi satu
baris teks yang agak ambigu dipisahkan (beda dengan `page.extract_table()`
yang akan mengembalikan struktur baris/kolom, tapi endpoint ini sengaja pakai
extract_text() sesuai kebutuhan chunking teks biasa).
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pdfplumber
from google import genai
from google.genai import types

CHUNK_SIZE = 600
CHUNK_OVERLAP_RATIO = 0.15
MIN_CHUNK_LENGTH = 50

# Heading bernomor seperti "3. Garansi Produk" di awal baris.
SECTION_HEADING_PATTERN = re.compile(r"(?m)^(\d+\.\s.+)$")

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
FAQ_PDF_PATH = DOCS_DIR / "faq-produk.pdf"

CHROMA_COLLECTION_NAME = "faq_knowledge_base"

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_OUTPUT_DIMENSIONALITY = 768
EMBEDDING_TIMEOUT_SECONDS = 30

# Hasil kalibrasi lewat scripts/calibrate_threshold.py (gemini-embedding-001,
# cosine similarity dari vektor L2-normalized): skor top-1 untuk 6 query dalam
# cakupan FAQ ada di rentang 0.7223-0.7494, sedangkan 6 query di luar cakupan
# ada di rentang 0.5966-0.6666 -- dua kelompok terpisah bersih (gap ~0.056).
# Titik tengahnya dipakai sebagai batas: skor top-1 di bawah ini dianggap
# query di luar cakupan knowledge base (chunk yang ketemu tidak relevan).
RETRIEVAL_THRESHOLD = 0.69

_CHROMA_WORKER_PATH = Path(__file__).resolve().parent / "_chroma_worker.py"


def _call_chroma_worker(command: str, payload: dict) -> dict:
    """Jalankan operasi vector-index chromadb (upsert/get_all/count) di
    subprocess terpisah lewat _chroma_worker.py -- lihat docstring file itu
    untuk alasannya (bentrok native library kalau digabung 1 proses dengan
    google-genai)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = Path(tmp_dir) / "input.json"
        output_path = Path(tmp_dir) / "output.json"
        input_path.write_text(json.dumps(payload), encoding="utf-8")

        subprocess.run(
            [sys.executable, str(_CHROMA_WORKER_PATH), command, str(input_path), str(output_path)],
            check=True,
        )

        return json.loads(output_path.read_text(encoding="utf-8"))


def _extract_text(file_path: str) -> str:
    with pdfplumber.open(file_path) as pdf:
        pages_text = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages_text)


def chunk_document(file_path: str, chunk_size: int = CHUNK_SIZE) -> list[dict]:
    """Ekstrak teks PDF lalu pecah jadi chunk ~chunk_size karakter dengan
    overlap 15%. Chunk yang lebih pendek dari MIN_CHUNK_LENGTH dibuang.

    Return list of {"chunk_id": "<source>::0001", "text": str, "source": str}.
    chunk_id deterministik & berurutan berdasarkan urutan chunk yang lolos filter.
    """
    source = Path(file_path).name
    full_text = _extract_text(file_path)

    overlap = int(chunk_size * CHUNK_OVERLAP_RATIO)
    step = chunk_size - overlap
    text_length = len(full_text)

    chunks: list[dict] = []
    seq = 1
    start = 0

    while start < text_length:
        end = start + chunk_size
        chunk_text = full_text[start:end].strip()

        if len(chunk_text) >= MIN_CHUNK_LENGTH:
            chunks.append(
                {
                    "chunk_id": f"{source}::{seq:04d}",
                    "text": chunk_text,
                    "source": source,
                }
            )
            seq += 1

        if end >= text_length:
            break
        start += step

    return chunks


RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _merge_splits(splits: list[str], separator: str, chunk_size: int, overlap: int) -> list[str]:
    """Gabung unit-unit kecil (baris/kalimat/kata) jadi chunk ~chunk_size,
    dengan overlap: beberapa unit terakhir dibawa lagi ke awal chunk berikutnya."""
    sep_len = len(separator)
    chunks: list[str] = []
    window: list[str] = []
    window_len = 0

    for piece in splits:
        piece_len = len(piece)
        extra = sep_len if window else 0

        if window and window_len + extra + piece_len > chunk_size:
            chunks.append(separator.join(window).strip())

            # Slide window mundur untuk overlap: buang unit dari depan sampai
            # sisa panjang window <= overlap.
            while window and window_len > overlap:
                removed = window.pop(0)
                window_len -= len(removed) + (sep_len if window else 0)

        window.append(piece)
        window_len += piece_len + (sep_len if len(window) > 1 else 0)

    if window:
        chunks.append(separator.join(window).strip())

    return [c for c in chunks if c]


def _split_text_recursive(text: str, separators: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Pecah `text` pakai separator pertama yang menghasilkan potongan valid;
    potongan yang masih lebih besar dari chunk_size di-split lagi rekursif
    pakai separator berikutnya yang lebih halus (paragraf -> baris -> kalimat -> kata)."""
    separator = separators[0]
    remaining_separators = separators[1:]
    splits = text.split(separator) if separator else list(text)

    good_splits: list[str] = []
    final_chunks: list[str] = []

    for piece in splits:
        if len(piece) < chunk_size:
            good_splits.append(piece)
            continue

        if good_splits:
            final_chunks.extend(_merge_splits(good_splits, separator, chunk_size, overlap))
            good_splits = []

        if not remaining_separators:
            final_chunks.append(piece)
        else:
            final_chunks.extend(_split_text_recursive(piece, remaining_separators, chunk_size, overlap))

    if good_splits:
        final_chunks.extend(_merge_splits(good_splits, separator, chunk_size, overlap))

    return final_chunks


def recursive_chunk(file_path: str, chunk_size: int = CHUNK_SIZE) -> list[dict]:
    """Chunking bertingkat (recursive character splitter): coba pecah per
    paragraf dulu ("\\n\\n"), kalau satu paragraf masih lebih besar dari
    chunk_size baru turun ke separator lebih halus (baris -> kalimat -> kata).
    Unit-unit yang muat digabung lagi sampai mendekati chunk_size, dengan
    overlap 15%, sehingga tidak pernah memotong di tengah kata/kalimat secara
    sembarangan seperti fixed-size slicing biasa.

    Return list of {"chunk_id": "<source>::r0001", "text": str, "source": str}.
    """
    source = Path(file_path).name
    full_text = _extract_text(file_path)
    overlap = int(chunk_size * CHUNK_OVERLAP_RATIO)

    raw_chunks = _split_text_recursive(full_text, RECURSIVE_SEPARATORS, chunk_size, overlap)

    chunks: list[dict] = []
    seq = 1
    for raw in raw_chunks:
        text = raw.strip()
        if len(text) >= MIN_CHUNK_LENGTH:
            chunks.append({"chunk_id": f"{source}::r{seq:04d}", "text": text, "source": source})
            seq += 1

    return chunks


def split_by_section(file_path: str) -> list[dict]:
    """Pecah dokumen berdasarkan heading bernomor (mis. "3. Garansi Produk"),
    bukan berdasarkan jumlah karakter tetap. Tiap section jadi 1 chunk utuh,
    apa pun panjangnya -- cocok buat membandingkan dengan fixed-size chunking.

    Return list of {"chunk_id": "<source>::section-N", "text": str, "source": str}.
    Teks sebelum heading pertama (mis. judul dokumen) masuk ke "section-0".
    """
    source = Path(file_path).name
    full_text = _extract_text(file_path)

    headings = list(SECTION_HEADING_PATTERN.finditer(full_text))
    if not headings:
        return chunk_document(file_path)

    sections: list[dict] = []

    preamble = full_text[: headings[0].start()].strip()
    if len(preamble) >= MIN_CHUNK_LENGTH:
        sections.append({"chunk_id": f"{source}::section-0", "text": preamble, "source": source})

    for i, heading in enumerate(headings, start=1):
        section_start = heading.start()
        section_end = headings[i].start() if i < len(headings) else len(full_text)
        section_text = full_text[section_start:section_end].strip()

        if len(section_text) >= MIN_CHUNK_LENGTH:
            sections.append({"chunk_id": f"{source}::section-{i}", "text": section_text, "source": source})

    return sections


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = sum(x * x for x in vector) ** 0.5
    if norm == 0:
        return vector
    return [x / norm for x in vector]


def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    """Embed sejumlah teks lewat gemini-embedding-001 (package google-genai).
    Tiap panggilan pakai timeout 30 detik & vektor hasil di-L2-normalize manual
    (gemini-embedding-001 dengan output_dimensionality != default tidak
    otomatis ter-normalize dari API-nya, beda dengan dimensi default 3072)."""
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=EMBEDDING_OUTPUT_DIMENSIONALITY,
        http_options=types.HttpOptions(timeout=EMBEDDING_TIMEOUT_SECONDS * 1000),
    )

    embeddings: list[list[float]] = []
    for i, text in enumerate(texts, start=1):
        print(
            f"[embed_texts] - [{i}/{len(texts)}] memanggil {EMBEDDING_MODEL} "
            f"(task_type={task_type}, timeout {EMBEDDING_TIMEOUT_SECONDS}s)..."
        )
        response = client.models.embed_content(model=EMBEDDING_MODEL, contents=text, config=config)
        raw_vector = response.embeddings[0].values
        embeddings.append(_l2_normalize(raw_vector))

    return embeddings


def index_chunks(chunks: list[dict], department: str = "general") -> None:
    """Embed tiap chunk (task_type=retrieval_document) lalu upsert ke koleksi
    Chroma "faq_knowledge_base", id=chunk_id, metadata {source, department}."""
    if not chunks:
        return

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts, task_type="retrieval_document")

    _call_chroma_worker(
        "upsert",
        {
            "ids": [chunk["chunk_id"] for chunk in chunks],
            "embeddings": embeddings,
            "documents": texts,
            "metadatas": [{"source": chunk["source"], "department": department} for chunk in chunks],
        },
    )


def get_collection_count() -> int:
    return _call_chroma_worker("count", {})["count"]


def search_similar(query: str, top_k: int = 3, department_filter: str | None = None) -> list[dict]:
    """Cari chunk paling mirip dengan `query` di collection faq_knowledge_base.
    Embed query dengan task_type=retrieval_query (normalisasi L2 sama seperti
    dokumen), lalu bandingkan ke semua embedding tersimpan. Return teks chunk,
    source, dan skor mentah (cosine similarity, makin besar makin mirip; kedua
    vektor sudah L2-normalized jadi cosine similarity == dot product) --
    bukan jawaban LLM.

    `department_filter` (mis. "general") diterapkan sebagai filter `where`
    native ChromaDB (bukan filter manual di Python) -- lihat _chroma_worker.py
    command "get_all". None berarti tidak difilter (semua department).

    Catatan implementasi: idealnya similarity search ini pakai `collection.query()`
    (ANN search native Chroma/HNSW) langsung di proses ini, tapi di environment
    ini (Python 3.14, di luar target 3.11+ project) menggabungkan google-genai
    dengan operasi native chromadb yang menyentuh vektor (upsert/get-embeddings
    /query) DALAM SATU PROSES YANG SAMA segfault (lihat _chroma_worker.py).
    Jadi pengambilan data mentah dari Chroma (termasuk filter `where`-nya, yang
    tetap dieksekusi oleh chromadb sendiri lewat collection.get(where=...))
    dijalankan di subprocess terpisah (_call_chroma_worker), lalu similarity
    dihitung manual (brute-force cosine) di sini -- untuk ukuran knowledge base
    sekecil ini tidak masalah dari sisi performa. Di environment Python
    3.11/3.12 yang kompatibel, boleh disederhanakan balik ke `collection.query()`
    in-process (where filter tetap dipakai sama)."""
    print(f"[search_similar] - embed query \"{query[:80]}\" (task_type=retrieval_query)...")
    query_embedding = embed_texts([query], task_type="retrieval_query")[0]

    where = {"department": department_filter} if department_filter else None
    stored = _call_chroma_worker("get_all", {"where": where})

    scored: list[dict] = []
    for text, metadata, embedding in zip(stored["documents"], stored["metadatas"], stored["embeddings"]):
        score = sum(q * d for q, d in zip(query_embedding, embedding))
        scored.append({"text": text, "source": metadata.get("source"), "score": score})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print("=" * 70)
    print("TEKS HASIL EKSTRAKSI PENUH (pdfplumber, extract_text)")
    print("=" * 70)
    full_text = _extract_text(str(FAQ_PDF_PATH))
    print(full_text)

    print("=" * 70)
    print("CHUNKING")
    print("=" * 70)
    chunks = chunk_document(str(FAQ_PDF_PATH))
    print(f"Jumlah chunk: {len(chunks)}")

    print("\n--- Chunk pertama ---")
    print(chunks[0])

    print("\n--- Chunk yang memuat tabel garansi ---")
    for chunk in chunks:
        if "Garansi" in chunk["text"] or "garansi" in chunk["text"]:
            print(chunk)
            print()

    print("=" * 70)
    print("INDEXING KE CHROMADB")
    print("=" * 70)
    index_chunks(chunks)
    print(f"Jumlah dokumen di collection '{CHROMA_COLLECTION_NAME}': {get_collection_count()}")
