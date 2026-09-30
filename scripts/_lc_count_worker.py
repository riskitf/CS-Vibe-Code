"""Worker TERPISAH (subprocess): hitung jumlah dokumen di sebuah collection
ChromaDB. Sengaja HANYA import chromadb (tidak pernah library genai apa pun)
supaya aman dijalankan setelah _lc_store_worker.py -- lihat catatan di sana.

argv: <persist_directory> <collection_name>
"""

import sys

import chromadb


def main() -> None:
    persist_directory = sys.argv[1]
    collection_name = sys.argv[2]

    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(collection_name)
    print(collection.count())


if __name__ == "__main__":
    main()
