"""Worker TERPISAH (dijalankan sebagai subprocess) khusus untuk operasi
chromadb yang menyentuh vector index (upsert, get dengan embeddings).

Kenapa perlu proses terpisah: di environment ini (Python 3.14, google-genai +
chromadb), sekadar meng-import google.genai lalu memanggil operasi native
chromadb yang menyentuh embedding (collection.get(include=["embeddings"]),
collection.upsert(), collection.query()) dalam proses yang SAMA menyebabkan
segfault (native library onnxruntime/HNSW milik chromadb bentrok dengan
grpc/protobuf yang dibawa google-genai). Operasi ringan seperti
get_or_create_collection() atau count() aman; yang tidak aman adalah operasi
yang menyentuh data vektornya. Worker ini sengaja HANYA meng-import chromadb
(tidak pernah google-genai), dipanggil lewat subprocess dari rag.py, tukar
data lewat file JSON supaya kedua library native itu tidak pernah hidup
bersamaan dalam satu proses Python.
"""

import json
import sys
from pathlib import Path

import chromadb

CHROMA_DIR = Path(__file__).resolve().parents[2] / ".chroma"
CHROMA_COLLECTION_NAME = "faq_knowledge_base"


def main() -> None:
    command = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    with open(input_path, encoding="utf-8") as f:
        payload = json.load(f)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)

    if command == "upsert":
        collection.upsert(
            ids=payload["ids"],
            embeddings=payload["embeddings"],
            documents=payload["documents"],
            metadatas=payload["metadatas"],
        )
        result = {"count": collection.count()}

    elif command == "get_all":
        where = payload.get("where") or None
        data = collection.get(where=where, include=["embeddings", "documents", "metadatas"])
        result = {
            "ids": list(data["ids"]),
            "documents": list(data["documents"]),
            "metadatas": list(data["metadatas"]),
            "embeddings": [list(vec) for vec in data["embeddings"]],
        }

    elif command == "count":
        result = {"count": collection.count()}

    else:
        raise ValueError(f"Unknown command: {command}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
