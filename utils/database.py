from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

def create_db(chunks, persist_directory="./vector_db"):
    """
    Create a Chroma vector database from the provided document chunks.
    """
    print("Creating vector database...")
    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

    print("Initializing Chroma vector store...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("Persisting vector database...")
    print(f"Database created and persisted at {persist_directory}.")
    return db