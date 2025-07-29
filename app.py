import streamlit as st
import os
from config import GROQ_API_KEY, EMBEDDING_MODEL_NAME, LANCEDB_PATH, LANCEDB_TABLE
from db.ingestor import Ingestor
from retrieval.retriever import Retriever
from llm.conversational import get_conversational_answer
import time
import pandas as pd

# Page configuration
st.set_page_config(page_title=" Hybrid RAG Chatbot", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Hybrid RAG Chatbot</h1>", unsafe_allow_html=True)
st.caption("Ask questions based on your document using Hybrid Search + LLM")

# Session state init
if "db_initialized" not in st.session_state:
    st.session_state["db_initialized"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# File upload section
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Supported: PDF, TXT, DOCX, or MD", type=["pdf", "txt", "docx", "md"])
    if uploaded_file and not st.session_state["db_initialized"]:
        os.makedirs("rag_chatbot/data", exist_ok=True)
        file_path = f"rag_chatbot/data/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        with st.spinner("🔄 Processing & indexing..."):
            ingestor = Ingestor(db_path=LANCEDB_PATH, table_name=LANCEDB_TABLE)
            ingestor.run(file_path)
            st.session_state["db_initialized"] = True
        st.success("✅ Document indexed! Start chatting below 👇")

# Main chat interface
if st.session_state["db_initialized"]:
    retriever = Retriever(db_path=LANCEDB_PATH, table_name=LANCEDB_TABLE, embedding_model=EMBEDDING_MODEL_NAME)
    st.subheader("💬 Ask Anything")
    
    # Chat history display
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Input box
    user_input = st.chat_input("Type your question here...")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.spinner(" Retrieving context..."):
            results = retriever.retrieve(user_input, mode="hybrid", k=6)
            top_texts = list(results["text"])
        with st.spinner(" Generating response..."):
            answer = get_conversational_answer(top_texts, user_input, GROQ_API_KEY)
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.rerun()
else:
    st.info("☝️ Upload a document to begin exploring it using AI!")
    