"""Worker TERPISAH (subprocess): bangun retriever dari vectorstore Chroma
yang sudah ada via .as_retriever(search_kwargs={"k": ...}) dan invoke satu
query -- TAPI sengaja HANYA import chromadb/langchain_chroma, TIDAK PERNAH
library genai apa pun.

Kenapa: retriever.invoke() secara internal memanggil embedding_function lalu
collection.query() (ANN search chromadb). Kalau embedding_function itu
GoogleGenerativeAIEmbeddings (butuh genai), kombinasi genai+chromadb dalam
satu proses segfault di environment Python 3.14 ini (root cause sama seperti
_lc_store_worker.py). Solusinya: query embedding-nya dihitung DI LUAR (proses
lain yang cuma genai, tidak pernah chromadb), lalu dioper ke worker ini lewat
input JSON. Worker ini bungkus vektor itu jadi "embedding_function" palsu
(cuma return vektor yang sudah dihitung, embed_query()-nya tidak benar-benar
manggil API apa pun) supaya vectorstore.as_retriever().invoke(query) tetap
dipakai apa adanya dan collection.query() chromadb-nya jalan di proses yang
bersih dari genai -> tidak segfault.

argv: <input_json_path> <output_json_path>
input JSON: {"persist_directory": str, "collection_name": str, "k": int,
             "query": str, "query_embedding": list[float]}
"""

import json
import sys

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


class _PrecomputedEmbeddings(Embeddings):
    """Embeddings palsu: embed_query() cuma return vektor yang sudah dihitung
    di proses lain, tidak pernah memanggil API/model apa pun -- supaya proses
    ini tidak perlu import library genai sama sekali."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector


def main() -> None:
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(input_path, encoding="utf-8") as f:
        payload = json.load(f)

    fake_embeddings = _PrecomputedEmbeddings(payload["query_embedding"])

    vectorstore = Chroma(
        persist_directory=payload["persist_directory"],
        collection_name=payload["collection_name"],
        embedding_function=fake_embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": payload["k"]})
    results = retriever.invoke(payload["query"])

    output = [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f)

    print("RETRIEVE_WORKER_DONE")


if __name__ == "__main__":
    main()
