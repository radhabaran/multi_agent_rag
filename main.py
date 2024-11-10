# main.py
import streamlit as st
import pandas as pd

# LangChain imports
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI

# Local imports
from vector_store import create_vector_store
from agents import classify_query, access_check, retrieve_data, generate_response, llm, tools  # Import tools


# Initialize vector stores (only once, using st.session_state)
if "finance_vectorstore" not in st.session_state:
    st.session_state.finance_vectorstore = create_vector_store("./data/finance_data.pdf", "finance_data")
    st.session_state.public_vectorstore = create_vector_store("./data/public_data.pdf", "public_data")

finance_vectorstore = st.session_state.finance_vectorstore
public_vectorstore = st.session_state.public_vectorstore

def rag_pipeline(query: str, username: str):
    """Main RAG pipeline."""
    query_type = classify_query(query)
    user_role = access_check(username)

    if query_type == "finance" and user_role != "admin":
        return "Access denied: Insufficient permissions for company confidential data."

    vectorstore = finance_vectorstore if query_type == "finance" else public_vectorstore
    # augmented_prompt = retrieve_data(query, vectorstore)
    # final_response = generate_response(augmented_prompt)

    # Use LangChain agent (initialize within the pipeline)
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True) # Verbose for debugging
    final_response = agent.run(query=query, vectorstore=vectorstore) # Pass vectorstore to the agent

    return final_response


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "username" not in st.session_state:
        st.session_state.username = ""

# Step 9: Streamlit UI
def main():
    """Streamlit web application."""
    st.title("RAG System")

    # Initialize session state
    initialize_session_state()
    
    # User authentication
    # if "username" not in st.session_state:
    #     st.session_state.username = None

    if st.session_state.username == '':
        auth_option = st.radio("Login or Signup:", ("Login", "Signup"))
        if auth_option == "Login":
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                # Authentication logic (check against users.csv)
                try:
                    users = pd.read_csv("users.csv")
                    user_data = users[users["username"] == username]
                    if not user_data.empty and user_data["password"].iloc[0] == password:
                        st.session_state.username = username
                        st.success("Logged in!")
                    else:
                        st.error("Incorrect username or password.")
                except FileNotFoundError:
                    st.error("User database not found.")
        elif auth_option == "Signup":
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")

            user_type = st.selectbox("User Type", ["admin", "user"])

            if st.button("Signup"):

                try:
                    users = pd.read_csv("users.csv")

                except FileNotFoundError:
                    users = pd.DataFrame(columns=["username", "password", "role"])
                new_user = pd.DataFrame({"username": [new_username], "password": [new_password], "role": [user_type]})
                users = pd.concat([users, new_user], ignore_index=True)
                users.to_csv("users.csv", index=False)
                st.success("Signup successful! Please login.")

    else:
        # User is logged in
        st.write(f"Welcome, {st.session_state.username}!")
        query = st.text_input("Enter your query:")
        if st.button("Submit"):
            response = rag_pipeline(query, st.session_state.username)
            st.write("Response:")
            st.write(response)
        if st.button("Logout"):
            st.session_state.username = None
            st.rerun()


if __name__ == "__main__":
    main()