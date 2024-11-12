import sqlite3
import os
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
from langchain.chat_models import ChatOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['OPENAI_API_KEY'] = os.getenv("OA_API")  # Or set directly
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

def create_vector_store(file_path: str, collection_name: str):
    """Creates and persists a Chroma vector store."""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        # Initialize embeddings outside try-except
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        persist_directory = f"./data/{collection_name}"

        logger.info(f"Creating new vector store for {collection_name}")
            
        # Select appropriate loader based on file type
        loader = PyPDFLoader(str(file_path))
            
        # Load documents
        documents = loader.load()
        logger.debug(f"Loaded {len(documents)} documents")

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=20,  # Added overlap
                length_function=len,
                add_start_index=True,
            )
        splits = text_splitter.split_documents(documents)  # Use split_documents instead of split_text
        logger.debug(f"Created {len(splits)} splits")

            # Create and persist vectorstore
        vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
        vectorstore.persist()
        logger.info(f"Created and persisted new vector store for {collection_name}")
            
        return vectorstore

    except Exception as e:
        logger.error(f"Error in create_vector_store: {str(e)}", exc_info=True)
        return None

def main():
    """Main function."""
    file_path1 = f"/workspaces/multi_agent_rag/docs/finance_data.pdf"
    create_vector_store(file_path1, 'finance')

    file_path2 = f"/workspaces/multi_agent_rag/docs/public_data.pdf"
    create_vector_store(file_path2, 'public')

if __name__ == "__main__":
    main()