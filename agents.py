# agents.py

import sqlite3
from packaging import version
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import os
import logging
import openai
import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough 
from typing import List, Dict, Any
import pandas as pd
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pathlib import Path
import getpass  # For secure password input during signup

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()  # Load environment variables from .env file

# Global context for storing vectorstore
retrieval_context = {'vectorstore': None}

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


def retrieve_and_generate(query: str) -> str:
    try:
        # Step1 : Retrieve documents
        vectorstore = retrieval_context['vectorstore']
    
        if vectorstore is None:
            return "Error: Vectorstore not initialized!"

        retriever = vectorstore.as_retriever(search_type="mmr", k=3)
        # Get documents directly from retriever
        docs = retriever.invoke(query)
            
       # Extract and combine the content from documents
        context = "\n".join(
            doc[0] if isinstance(doc, tuple) else doc.page_content 
            for doc in docs
        )
        
        # Step 2: Format prompt and generate response
        # Define template
        template = """Respond to the query using the provided context. If the answer is not contained
        within the context, say 'I don't know.' Be concise and extract relevant information from 
        the context.

        Query: {query}

        Context: {context}

        Answer:"""

        # Format the template with actual values
        formatted_prompt = template.format(
            query=query,
            context=context
        )
        
        # Step 3: Generate response
        messages = [
            {
                "role": "user",
                "content": formatted_prompt
            }
        ]
        
        response = llm.invoke(messages)
        
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, dict) and 'content' in response:
            return response['content']
        else:
            return str(response)
            
    except Exception as e:
        logger.error(f"Error in retrieve_and_generate: {str(e)}")
        return f"Error processing query: {str(e)}"



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
        name="Knowledge Base QA",
        func=retrieve_and_generate,
        description="Use this tool to pass the user query to the appropriate knowledge base. This tool will retrieve relevant information and generate a helpful response based on the context.",
        return_direct = True
        ),
    ]

# Initialize the agent once
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    handle_parsing_errors=True
)