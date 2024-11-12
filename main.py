# main.py
import sqlite3
import streamlit as st
import pandas as pd
import os
import sys

try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

from pathlib import Path
import chromadb
from typing import Tuple
import logging
logger = logging.getLogger(__name__)

# LangChain imports
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Local imports
from agents import agent, retrieval_context, classify_query, access_check

# Ensure data directory exists
os.makedirs("./data", exist_ok=True)

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "username" not in st.session_state:
        st.session_state.username = ""

    if "query" not in st.session_state:
        st.session_state.query = ""

    if "auth_option" not in st.session_state:
        st.session_state.auth_option = "Login"

    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if "vectorstores_initialized" not in st.session_state:
        st.session_state.vectorstores_initialized = False
        
    # Only initialize vectorstores once
    if not st.session_state.vectorstores_initialized:
        initialize_vector_stores()
        st.session_state.vectorstores_initialized = True

def initialize_vector_stores():
    """Initialize vector stores by loading existing ones."""
    try:
        # Initialize embeddings
        embeddings = OpenAIEmbeddings()

        # Load existing vector stores
        st.session_state.finance_vectorstore = Chroma(
            persist_directory="./data/finance",
            collection_name="finance",
            embedding_function=embeddings
        )
        
        st.session_state.public_vectorstore = Chroma(
            persist_directory="./data/public",
            collection_name="public",
            embedding_function=embeddings
        )

    except Exception as e:
        logger.error(f"Failed to initialize vectorstores: {str(e)}")
        st.session_state.finance_vectorstore = None
        st.session_state.public_vectorstore = None


def rag_pipeline(query: str, username: str):
    """Main RAG pipeline."""

    # Classify query using the existing function
    query_type = classify_query(query)
    
    
    # Check user permissions
    user_role = access_check(username)

    if query_type == "finance" and user_role != "admin":
        return "Access denied: Insufficient permissions for company confidential data."

    # Select appropriate vectorstore
    vectorstore = (st.session_state.finance_vectorstore 
                  if query_type == "finance" 
                  else st.session_state.public_vectorstore)

    # Set the vectorstore in the global context for the agent to use
    retrieval_context['vectorstore'] = vectorstore
        
    # Invoke the agent with just the query
    response = agent.invoke({"input": query})
    return response["output"]


def handle_authentication(username: str, password: str) -> bool:
    """Handle user authentication."""
    try:
        users = pd.read_csv("users.csv")
        user_data = users[users["username"] == username]
        if not user_data.empty and user_data["password"].iloc[0] == password:
            return True
        return False
    except FileNotFoundError:
        st.error("User database not found.")
        return False


def handle_signup(username: str, password: str, user_type: str) -> bool:
    """Handle user signup."""
    if not username or not password:
        st.error("Username and password are required.")
        return False

    try:
        users = pd.read_csv("users.csv")
        if username in users["username"].values:
            st.error("Username already exists.")
            return False
    except FileNotFoundError:
        users = pd.DataFrame(columns=["username", "password", "role"])

    new_user = pd.DataFrame({
        "username": [username],
        "password": [password],
        "role": [user_type]
    })
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv("users.csv", index=False)
    return True


def render_auth_forms():
    """Render authentication forms."""
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"Current option: {st.session_state.auth_option}")
    
    with col2:
        if st.button("Switch to Login" if st.session_state.auth_option == "Sign Up" else "Switch to Sign Up"):
            st.session_state.auth_option = "Sign Up" if st.session_state.auth_option == "Login" else "Login"
            st.rerun()

    with st.form(key="auth_form"):
        username = st.text_input("Username").strip()
        password = st.text_input("Password", type="password").strip()
        
        if st.session_state.auth_option == "Sign Up":
            role = st.selectbox("Role", ["user", "admin"])
            submit_button = st.form_submit_button("Sign Up")
            
            if submit_button:
                if not username or not password:
                    st.error("Username and password are required!")
                elif handle_signup(username, password, role):
                    st.success("Sign up successful! Please login.")
                    st.session_state.auth_option = "Login"
                    st.rerun()
                else:
                    st.error("Username already exists!")
        else:
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if not username or not password:
                    st.error("Username and password are required!")
                elif handle_authentication(username, password):
                    st.session_state.username = username
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials!")


def render_chat_interface():
    """Render chat interface."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"Logged in as: {st.session_state.username}")
    
    with col2:
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        prompt = prompt.strip()
        if prompt:  # Only process non-empty prompts
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = rag_pipeline(prompt, st.session_state.username)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


# Steps for Streamlit UI
def main():
    """Main function."""
    st.set_page_config(page_title="Secure RAG Chatbot", page_icon="🤖", layout="wide")
    st.title("🤖 Secure RAG Chatbot")
    
    # Initialize users.csv if it doesn't exist
    if not Path("users.csv").exists():
        pd.DataFrame(columns=["username", "password", "role"]).to_csv("users.csv", index=False)
    
    initialize_session_state()
    
    if not st.session_state.authenticated:
        render_auth_forms()
    else:
        render_chat_interface()

if __name__ == "__main__":
    main()