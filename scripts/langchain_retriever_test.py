"""Bangun retriever dari vectorstore Chroma yang sudah dibuat di
scripts/langchain_chunk_faq.py, pakai .as_retriever(search_kwargs={"k": 3}),
lalu test dengan satu query dan print 3 hasil teratas + 100 karakter pertama
isinya.

Jalankan: PYTHONPATH=. python scripts/langchain_retriever_test.py
(Pastikan scripts/langchain_chunk_faq.py sudah pernah dijalankan dulu supaya
vectorstore-nya ada.)
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

from langchain_chunk_faq import (  # noqa: E402
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    SCRIPTS_DIR,
)

QUERY = "berapa lama garansi headset"
TOP_K = 3


def main() -> None:
    # Hitung embedding query di proses ini (cuma genai, belum sentuh chromadb
    # sama sekali) -- baru diteruskan ke worker terpisah yang menjalankan
    # .as_retriever().invoke() di proses yang bersih dari genai. Lihat
    # docstring _lc_retrieve_worker.py untuk kenapa perlu dipisah begini.
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.environ["GEMINI_API_KEY"],
    )
    query_embedding = embeddings.embed_query(QUERY)

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = Path(tmp_dir) / "retrieve_input.json"
        output_path = Path(tmp_dir) / "retrieve_output.json"

        input_path.write_text(
            json.dumps(
                {
                    "persist_directory": CHROMA_PERSIST_DIR,
                    "collection_name": CHROMA_COLLECTION_NAME,
                    "k": TOP_K,
                    "query": QUERY,
                    "query_embedding": query_embedding,
                }
            ),
            encoding="utf-8",
        )

        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "_lc_retrieve_worker.py"), str(input_path), str(output_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        if "RETRIEVE_WORKER_DONE" not in result.stdout:
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError("Retriever worker tidak menghasilkan output yang diharapkan")

        results = json.loads(output_path.read_text(encoding="utf-8"))

    print(f'Query: "{QUERY}"')
    print(f"Jumlah hasil (k={TOP_K}): {len(results)}")
    print()
    for i, r in enumerate(results, start=1):
        preview = r["content"][:100].replace("\n", " ")
        print(f"[{i}] source={r['metadata'].get('source')}  page={r['metadata'].get('page')}")
        print(f"    {preview}...")
        print()


if __name__ == "__main__":
    main()
