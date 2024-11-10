# agents.py
import os
import openai
import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from typing import List, Dict, Any
import pandas as pd
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import getpass  # For secure password input during signup

load_dotenv()  # Load environment variables from .env file

os.environ['OPENAI_API_KEY'] = os.getenv("OA_API")  # Or set directly
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

def classify_query(query: str) -> str:
    """Classifies the query as 'finance' or 'public'."""
    # Basic classification – replace with more sophisticated logic if needed
    if "finance" in query.lower() or "financial" in query.lower():
        return "finance"
    return "public"


def access_check(username: str) -> str:
    """Checks user access rights."""
    try:
        users = pd.read_csv("users.csv")
        user_role = users[users["username"] == username]["role"].iloc[0]
        return user_role
    except (FileNotFoundError, IndexError):
        return "guest" # Default to guest if user data is unavailable


def retrieve_data(query: str, vectorstore: Chroma):
    """Retrieves relevant documents and augments the prompt."""
    # Use MMR for diverse results (experiment with different fetch_k values)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    return prompt


def generate_response(prompt: str) -> str:
    """Generates the final response using the LLM."""
    response = llm(prompt)
    return response

# Create tools for the agent
#
# return_direct=False: 
# The output will be passed back to the agent for further analysis 
# or to potentially use other tools. This is appropriate for tools like "Data Retriever" 
# because we want the agent to analyze the retrieved data and potentially use it with other tools.
#
# return_direct=True: 
# The output will be returned directly to the user without further agent processing. 
# This makes sense for "Q&A Agent" as it generates the final response.

tools = [
    Tool(
        name="Data Retriever",
        func=retrieve_data,
        description="Retrieves relevant context from the vector store based on the query. Use this tool first to gather information.",
        return_direct = False
    ),
    Tool(
        name="Q&A Agent",
        func=generate_response,
        description="Generates natural language responses using the retrieved context and original question. Use this after getting context.",
        return_direct = True
    )
]