from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import config

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter


_embeddings = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    return _embeddings


def store_text_in_local_db(text: str, collection_name: str = "agentic_explainer"):
    """
    Chunks extracted text and stores it in a local ChromaDB directory.
    """
    if not text or not text.strip():
        print("No text found to store in Vector DB.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    print(f"--- SAVING TO LOCAL DB: Storing {len(chunks)} chunks in ChromaDB ---")

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=_get_embeddings(),
        collection_name=collection_name,
        persist_directory=config.CHROMA_DB_DIR,
    )
    return vectorstore


def query_vector_store(query: str, collection_name: str = "agentic_explainer", top_k: int = 3) -> str:
    """
    Retrieves the most relevant document chunks from the local ChromaDB.
    """
    print("--- QUERYING LOCAL DB: Searching for context ---")

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=_get_embeddings(),
        persist_directory=config.CHROMA_DB_DIR,
    )

    docs = vectorstore.similarity_search(query, k=top_k)

    if not docs:
        return "No relevant context found in the document."

    return "\n\n".join(doc.page_content for doc in docs)
