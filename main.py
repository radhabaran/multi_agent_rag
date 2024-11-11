# main.py
import streamlit as st
import pandas as pd
import os
from pathlib import Path

# LangChain imports
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

# Local imports
from vector_store import create_vector_store
from agents import (classify_query, access_check, retrieve_data, generate_response, 
                    llm, tools, agent)  # Import tools


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


def initialize_vector_stores():
    """Initialize vector stores if not already in session state."""
    if "finance_vectorstore" not in st.session_state:
        st.session_state.finance_vectorstore = create_vector_store(
            "./data/finance_data.pdf", "finance_data")
        st.session_state.public_vectorstore = create_vector_store(
            "./data/public_data.pdf", "public_data")

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

    # Use the pre-initialized agent from agents.py
    final_response = agent.run(query=query, vectorstore=vectorstore) # Pass vectorstore to the agent

    return final_response

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


# Steps for Streamlit UI
def main():
    """Streamlit web application."""
    st.title("Secure RAG System")

    # Initialize session state
    initialize_session_state()
    initialize_vector_stores()
    
    # Query input field (always visible)
    query = st.text_input("Enter your query:", key="query_input")

    # Authentication section
    if not st.session_state.username:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login Mode"):
                st.session_state.auth_option = "Login"
        with col2:
            if st.button("Signup Mode"):
                st.session_state.auth_option = "Signup"

        if st.session_state.auth_option == "Login":
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit_login = st.form_submit_button("Login")

                if submit_login:
                    if handle_authentication(username, password):
                        st.session_state.username = username
                        st.success("Logged in successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")

        else:  # Signup mode
            with st.form("signup_form"):
                new_username = st.text_input("New Username")
                new_password = st.text_input("New Password", type="password")
                user_type = st.selectbox("User Type", ["user", "admin"])
                submit_signup = st.form_submit_button("Signup")

                if submit_signup:
                    if handle_signup(new_username, new_password, user_type):
                        st.success("Signup successful! Please login.")
                        st.session_state.auth_option = "Login"
                        st.rerun()

    else:
        # User is logged in
        st.write(f"Logged in as: {st.session_state.username}")
        
        if query:  # Process query if input exists
            response = rag_pipeline(query, st.session_state.username)
            st.write("Response:")
            st.write(response)

        if st.button("Logout"):
            st.session_state.username = ""
            st.session_state.query = ""
            st.rerun()

if __name__ == "__main__":
    # Ensure the users.csv file exists
    if not Path("users.csv").exists():
        pd.DataFrame(columns=["username", "password", "role"]).to_csv("users.csv", index=False)
    main()