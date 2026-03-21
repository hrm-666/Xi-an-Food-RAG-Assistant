import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def _has_persisted_vector_db(persist_directory):
    if not os.path.isdir(persist_directory):
        return False

    return any(os.scandir(persist_directory))


def create_db(chunks, persist_directory="./vector_db"):
    """
    Load an existing Chroma vector database if available; otherwise create one.
    """
    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

    if _has_persisted_vector_db(persist_directory):
        print(f"Loading existing vector database from {persist_directory}...")
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
        print("Vector database loaded.")
        return db

    print("Creating vector database...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    print(f"Database created and persisted at {persist_directory}.")
    return db
