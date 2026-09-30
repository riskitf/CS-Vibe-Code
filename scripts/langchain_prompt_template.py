"""Prompt template RAG pakai ChatPromptTemplate (langchain_core.prompts),
dipakai bareng retriever di scripts/langchain_retriever_test.py.

Jalankan: PYTHONPATH=. python scripts/langchain_prompt_template.py
"""

from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    "Kamu adalah asisten customer support toko online. Jawab pertanyaan customer "
    "dalam bahasa Indonesia, singkat dan jelas.\n\n"
    "Jawab HANYA berdasarkan konteks yang diberikan di bawah ini. Jangan menebak. "
    "Kalau jawabannya tidak ada di konteks, jawab persis: \"Maaf, saya tidak tahu\".\n\n"
    "Konteks:\n{context}\n\n"
    "Pertanyaan: {question}\n\n"
    "Akhiri jawabanmu dengan sumber, format: (sumber: <nama_file>)."
)


if __name__ == "__main__":
    example_context = (
        "[1] (sumber: faq-produk.pdf)\n"
        "Kategori Produk Masa Garansi Cakupan\n"
        "Audio (headset, speaker) 12 bulan Cacat produksi & kerusakan komponen"
    )
    example_question = "berapa lama garansi headset"

    messages = RAG_PROMPT_TEMPLATE.format_messages(context=example_context, question=example_question)

    print("--- Prompt yang sudah diisi (contoh) ---")
    for message in messages:
        print(f"[{message.type}]")
        print(message.content)
