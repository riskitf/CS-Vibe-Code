"""Baca & potong docs/faq-produk.pdf, lalu embed & simpan ke ChromaDB --
semuanya pakai LangChain (PyPDFLoader, RecursiveCharacterTextSplitter,
GoogleGenerativeAIEmbeddings, Chroma), sebagai pembanding terhadap
implementasi manual di app/services/rag.py.

Catatan environment: embedding (GoogleGenerativeAIEmbeddings) + penyimpanan
(Chroma.from_documents) dijalankan lewat subprocess terpisah
(_lc_store_worker.py), karena menggabungkan chromadb dengan library berbasis
google-genai DALAM SATU PROSES menyebabkan segfault native di Python 3.14
(environment project ini, di luar target 3.11+) -- persis kasus yang sama
dengan app/services/rag.py + _chroma_worker.py. Data tetap tersimpan dengan
benar walau proses worker itu sendiri crash setelah selesai menulis; jumlah
dokumen final diverifikasi lewat proses terpisah lagi (_lc_count_worker.py)
yang cuma import chromadb, tidak akan crash.

Jalankan: PYTHONPATH=. python scripts/langchain_chunk_faq.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

SCRIPTS_DIR = Path(__file__).resolve().parent
FAQ_PDF_PATH = SCRIPTS_DIR.parent / "docs" / "faq-produk.pdf"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 90  # 15% dari 600

EMBEDDING_MODEL = "models/gemini-embedding-001"
CHROMA_PERSIST_DIR = str(SCRIPTS_DIR.parent / ".chroma_langchain")
CHROMA_COLLECTION_NAME = "faq_knowledged_base_lc"


def main() -> None:
    loader = PyPDFLoader(str(FAQ_PDF_PATH))
    documents = loader.load()
    print(f"Jumlah halaman terbaca: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    print(f"Jumlah chunk yang dihasilkan: {len(chunks)}")
    print()
    print("--- Isi chunk pertama ---")
    print(f"metadata: {chunks[0].metadata}")
    print(f"panjang: {len(chunks[0].page_content)} karakter")
    print("konten:")
    print(chunks[0].page_content)

    print()
    print("--- Embedding (gemini-embedding-001) & penyimpanan ke ChromaDB via Chroma.from_documents() ---")

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = Path(tmp_dir) / "store_input.json"
        input_path.write_text(
            json.dumps(
                {
                    "chunks": [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks],
                    "persist_directory": CHROMA_PERSIST_DIR,
                    "collection_name": CHROMA_COLLECTION_NAME,
                    "embedding_model": EMBEDDING_MODEL,
                    "api_key": os.environ["GEMINI_API_KEY"],
                }
            ),
            encoding="utf-8",
        )

        # Catatan: proses worker ini kadang segfault SETELAH datanya berhasil
        # ditulis ke ChromaDB (native library onnxruntime/HNSW milik chromadb
        # bentrok dengan library genai dalam satu proses, di environment
        # Python 3.14 ini) -- exit code / crash di sini TIDAK dianggap gagal.
        # Yang jadi acuan sukses adalah jumlah dokumen sungguhan yang
        # dikonfirmasi lewat _lc_count_worker.py (proses terpisah, aman).
        store_result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "_lc_store_worker.py"), str(input_path)],
            capture_output=True,
            text=True,
        )
        if store_result.returncode != 0:
            print(
                f"[info] Proses embedding/penyimpanan keluar dengan kode {store_result.returncode} "
                "(kemungkinan crash native library) -- memverifikasi hasil aktualnya lewat proses terpisah..."
            )

    count_result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "_lc_count_worker.py"), CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME],
        capture_output=True,
        text=True,
        check=True,
    )
    stored_count = int(count_result.stdout.strip())

    if stored_count < len(chunks):
        raise RuntimeError(
            f"Penyimpanan tidak lengkap: baru {stored_count} dari {len(chunks)} chunk tersimpan di ChromaDB."
        )

    print(f"Jumlah dokumen yang berhasil tersimpan di '{CHROMA_COLLECTION_NAME}': {stored_count}")


if __name__ == "__main__":
    main()
