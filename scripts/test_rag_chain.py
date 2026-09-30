"""Test rag_chain (scripts/langchain_rag_chain.py) dengan 2 pertanyaan,
panggil pakai .invoke().

Catatan environment PENTING: rag_chain.invoke(question) secara literal TIDAK
BISA dijalankan langsung dalam satu proses di environment ini (Python 3.14) --
retriever di dalam chain menyentuh chromadb native (collection.query()) dan
llm-nya (ChatGoogleGenerativeAI) menyentuh google.genai; kombinasi keduanya
DALAM SATU PROSES segfault (root cause sama yang berulang kali ditemukan di
scripts/_lc_store_worker.py & scripts/_lc_retrieve_worker.py).

Solusinya di sini: langkah retriever dijalankan lewat subprocess terpisah
yang sudah terbukti aman (_lc_retrieve_worker.py, chromadb-only), hasilnya
diformat pakai format_docs() yang SAMA PERSIS dengan yang dipakai rag_chain,
lalu "ekor" chain (RAG_PROMPT_TEMPLATE | llm | StrOutputParser()) -- objek
YANG SAMA PERSIS dari rag_chain, bukan dibuat ulang -- di-invoke() di proses
utama (genai-only, tidak pernah menyentuh chromadb). Secara fungsional ini
persis rag_chain.invoke(question), cuma langkah retriever-nya dieksekusi di
proses terisolasi supaya tidak segfault.

Jalankan: PYTHONPATH=. python scripts/test_rag_chain.py
"""

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

from langchain_chunk_faq import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, EMBEDDING_MODEL, SCRIPTS_DIR  # noqa: E402
from langchain_rag_chain import RAG_PROMPT_TEMPLATE, format_docs, llm  # noqa: E402

QUERIES = [
    "berapa lama garansi headset",
    "ada diskon member",
]
TOP_K = 3

# "Ekor" chain -- persis komponen prompt/llm/parser yang sama dengan rag_chain,
# tinggal tanpa segmen retriever (yang dieksekusi terpisah lewat subprocess).
answer_chain = RAG_PROMPT_TEMPLATE | llm | StrOutputParser()


@dataclass
class SimpleDoc:
    page_content: str
    metadata: dict


def retrieve_docs(query: str, k: int = TOP_K) -> list[SimpleDoc]:
    """Jalankan langkah retriever rag_chain lewat subprocess terisolasi
    (sama seperti scripts/langchain_retriever_test.py)."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.environ["GEMINI_API_KEY"],
    )
    query_embedding = embeddings.embed_query(query)

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = Path(tmp_dir) / "retrieve_input.json"
        output_path = Path(tmp_dir) / "retrieve_output.json"

        input_path.write_text(
            json.dumps(
                {
                    "persist_directory": CHROMA_PERSIST_DIR,
                    "collection_name": CHROMA_COLLECTION_NAME,
                    "k": k,
                    "query": query,
                    "query_embedding": query_embedding,
                }
            ),
            encoding="utf-8",
        )

        subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "_lc_retrieve_worker.py"), str(input_path), str(output_path)],
            capture_output=True,
            text=True,
        )

        if not output_path.exists():
            raise RuntimeError(f"Retriever gagal untuk query: {query!r}")

        raw_results = json.loads(output_path.read_text(encoding="utf-8"))

    return [SimpleDoc(page_content=r["content"], metadata=r["metadata"]) for r in raw_results]


def main() -> None:
    for query in QUERIES:
        print("=" * 70)
        print(f'Pertanyaan: "{query}"')

        docs = retrieve_docs(query)
        context = format_docs(docs)

        answer = answer_chain.invoke({"context": context, "question": query})

        print(f"Jawaban: {answer}")


if __name__ == "__main__":
    main()
