import streamlit as st
import os
import time
import pandas as pd
from config import GROQ_API_KEY, EMBEDDING_MODEL_NAME, LANCEDB_PATH, LANCEDB_TABLE
from db.ingestor import Ingestor
from retrieval.retriever import Retriever
from llm.conversational import get_conversational_answer
import os
from dotenv import load_dotenv

load_dotenv()  

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Page configuration
st.set_page_config(page_title="Hybrid RAG Chatbot", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Hybrid RAG Chatbot</h1>", unsafe_allow_html=True)
st.caption("Ask questions based on your document using Hybrid Search + LLM")

# Session state initialization
if "db_initialized" not in st.session_state:
    st.session_state["db_initialized"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "latency_logs" not in st.session_state:
    st.session_state["latency_logs"] = []

# Sidebar - Upload
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

    # Sidebar - Latency metrics
    if st.session_state["latency_logs"]:
        st.markdown("---")
        st.header("Latency Metrics")
        latest = st.session_state["latency_logs"][-1]
        latest_table = pd.DataFrame(
            {
                "Components": [
                    "Embedding",
                    "Dense Search",
                    "Sparse Search",
                    "Fusion",
                    "Retrieval",
                    "Generation",
                    "Total",
                ],
                "Time (s)": [
                    round(latest.get("embedding_time", 0), 3),
                    round(latest.get("dense_search_time", 0), 3),
                    round(latest.get("sparse_search_time", 0), 3),
                    round(latest.get("fusion_time", 0), 3),
                    round(latest.get("retrieval_time", 0), 3),
                    round(latest.get("generation_time", 0), 3),
                    round(latest.get("total_time", 0), 3),
                ],
            }
        )
        st.markdown("**Latest Query**")
        st.table(latest_table)

        df = pd.DataFrame(st.session_state["latency_logs"])
        avg_times = df.mean(numeric_only=True).to_dict()
        avg_table = pd.DataFrame(
            {
                "Components": [
                    "Embedding",
                    "Dense Search",
                    "Sparse Search",
                    "Fusion",
                    "Retrieval",
                    "Generation",
                    "Total",
                ],
                "Time (s)": [
                    round(avg_times.get("embedding_time", 0), 3),
                    round(avg_times.get("dense_search_time", 0), 3),
                    round(avg_times.get("sparse_search_time", 0), 3),
                    round(avg_times.get("fusion_time", 0), 3),
                    round(avg_times.get("retrieval_time", 0), 3),
                    round(avg_times.get("generation_time", 0), 3),
                    round(avg_times.get("total_time", 0), 3),
                ],
            }
        )
        st.markdown("**Average Latency (All queries)**")
        st.table(avg_table)

        # ---- All Queries Latency Log ----
        st.markdown("**🧾 All Queries**")
        full_logs_df = pd.DataFrame(st.session_state["latency_logs"])
        # Show selected columns for display
        display_logs = full_logs_df[
            [
                "embedding_time",
                "dense_search_time",
                "sparse_search_time",
                "fusion_time",
                "retrieval_time",
                "generation_time",
                "total_time",
            ]
        ].round(3)
        display_logs.index = [f"Query {i+1}" for i in range(len(display_logs))]
        st.dataframe(display_logs, use_container_width=True)
        # CSV data includes queries and responses
        logs_with_queries = full_logs_df[
            [
                "query",
                "response",
                "embedding_time",
                "dense_search_time",
                "sparse_search_time",
                "fusion_time",
                "retrieval_time",
                "generation_time",
                "total_time",
            ]
        ].round(3)
        csv = logs_with_queries.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Logs with Responses",
            data=csv,
            file_name="latency_logs_with_responses.csv",
            mime='text/csv'
        )

# Main chat interface
if st.session_state["db_initialized"]:
    retriever = Retriever(
        db_path=LANCEDB_PATH, table_name=LANCEDB_TABLE, embedding_model=EMBEDDING_MODEL_NAME
    )
    st.subheader("💬 Ask Anything")

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box for user queries
    user_input = st.chat_input("Type your question here...", key="main_chat_input")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})

        query_times = {}
        total_start = time.perf_counter()

        # --- Retrieval ---
        retrieve_start = time.perf_counter()
        with st.spinner("🔍 Retrieving context..."):
            results, timings_of_ind_comp = retriever.retrieve(user_input, mode="hybrid", k=10)
            top_texts = list(results["text"])
        retrieve_end = time.perf_counter()
        query_times["retrieval_time"] = (retrieve_end - retrieve_start)

        # Save individual component times
        query_times["embedding_time"] = timings_of_ind_comp.get("embedding_time", 0)
        query_times["dense_search_time"] = timings_of_ind_comp.get("dense_search_time", 0)
        query_times["sparse_search_time"] = timings_of_ind_comp.get("sparse_search_time", 0)
        query_times["fusion_time"] = timings_of_ind_comp.get("fusion_time", 0)

        # --- Generation ---
        gen_start = time.perf_counter()
        with st.spinner(" Generating response..."):
            answer = get_conversational_answer(top_texts, user_input, GROQ_API_KEY)
        gen_end = time.perf_counter()
        query_times["generation_time"] = (gen_end - gen_start)

        # --- Total Time ---
        total_end = time.perf_counter()
        query_times["total_time"] = (total_end - total_start)

        # --- Include query text and response
        query_times["query"] = user_input
        query_times["response"] = answer

        # Save latency info
        st.session_state["latency_logs"].append(query_times)

        # Store assistant response
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.rerun()
else:
    st.info("☝️ Upload a document to begin exploring it using AI!")