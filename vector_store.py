# vector_store.py

import sqlite3
from packaging import version
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import logging
from pathlib import Path
import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, TextLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vectorstore creation
def create_vector_store(file_path: str, collection_name: str):
    """Creates and persists a Chroma vector store."""
    try:
        # Load existing collection
        vectorstore = Chroma(persist_directory=f"./data/{collection_name}", embedding_function=OpenAIEmbeddings())
        print(f"Loaded existing vector store for {collection_name}")
        return vectorstore
    except:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Select appropriate loader based on file type
        if file_path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path))

        # Load and split documents
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(documents)

        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Use ada for embeddings
        vectorstore = Chroma.from_texts(texts, embeddings, collection_name=collection_name, persist_directory=f"./data/{collection_name}")
        vectorstore.persist()
        print(f"Created new vector store for {collection_name}")
        return vectorstore