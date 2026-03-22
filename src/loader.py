import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from src.cleaner import clean_documents

def load_tech_docs(directory):
    """
    Load technical documents from a specified directory.
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return []
    
    loader = DirectoryLoader(
        path=directory,
        glob="**/*.pdf",#解析目录下的所有pdf文件
        loader_cls=PyPDFLoader
    )

    print(f"Loading documents from {directory}...")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return clean_documents(documents)



