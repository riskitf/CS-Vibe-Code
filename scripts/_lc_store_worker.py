"""Worker TERPISAH (subprocess): jalankan Chroma.from_documents() dari
langchain_chroma dengan GoogleGenerativeAIEmbeddings. Diisolasi ke proses
sendiri karena kombinasi chromadb + library berbasis google-genai dalam satu
proses segfault di environment ini (Python 3.14) -- lihat catatan panjang di
app/services/_chroma_worker.py untuk detail root cause. Proses ini boleh
crash (segfault) SETELAH datanya berhasil ditulis; verifikasi jumlah dokumen
yang benar-benar tersimpan dilakukan lewat scripts/_lc_count_worker.py
(proses lain yang cuma import chromadb, tidak akan crash).

argv: <input_json_path>
input JSON: {"chunks": [{"page_content": str, "metadata": dict}, ...],
             "persist_directory": str, "collection_name": str,
             "embedding_model": str, "api_key": str}
"""

import json
import sys

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def main() -> None:
    input_path = sys.argv[1]
    with open(input_path, encoding="utf-8") as f:
        payload = json.load(f)

    documents = [Document(page_content=c["page_content"], metadata=c["metadata"]) for c in payload["chunks"]]

    embeddings = GoogleGenerativeAIEmbeddings(
        model=payload["embedding_model"],
        google_api_key=payload["api_key"],
    )

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=payload["persist_directory"],
        collection_name=payload["collection_name"],
    )

    print("STORE_WORKER_DONE")


if __name__ == "__main__":
    main()
