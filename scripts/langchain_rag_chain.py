"""Rangkai RAG chain lengkap pakai LCEL (operator |):
retriever | format_docs -> isi {context}, RunnablePassthrough -> isi
{question}, lalu prompt template (langchain_prompt_template.py),
ChatGoogleGenerativeAI (gemini-3.7-flash), StrOutputParser.

Jalankan: PYTHONPATH=. python scripts/langchain_rag_chain.py
"""

import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

from langchain_chunk_faq import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, EMBEDDING_MODEL  # noqa: E402
from langchain_prompt_template import RAG_PROMPT_TEMPLATE  # noqa: E402

CHAT_MODEL = "gemini-3.7-flash"


def format_docs(docs: list) -> str:
    """Gabungkan hasil retriever jadi teks bernomor [1], [2], dst, masing-masing
    disertai sumbernya, buat diisi ke placeholder {context} di prompt."""
    return "\n\n".join(
        f"[{i}] (sumber: {doc.metadata.get('source')})\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )


embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=os.environ["GEMINI_API_KEY"],
)

vectorstore = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    collection_name=CHROMA_COLLECTION_NAME,
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(
    model=CHAT_MODEL,
    google_api_key=os.environ["GEMINI_API_KEY"],
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT_TEMPLATE
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":
    print("rag_chain berhasil dirangkai:")
    print(rag_chain)
