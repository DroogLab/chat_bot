import streamlit as st
import os
import time
import pandas as pd
import re
from config import GROQ_API_KEY, EMBEDDING_MODEL_NAME, LANCEDB_PATH
from db.ingestor import Ingestor
from retrieval.retriever import Retriever
from llm.conversational import get_conversational_answer
from dotenv import load_dotenv
import lancedb

DATA_DIR = "rag_chatbot/data"

# below function is to tackle the inappropriate file name like DC 2.pdf (inbetween there is a space, which is replaced with _ )
def sanitize_table_name(name):
    name_no_ext = os.path.splitext(name)[0]
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', name_no_ext)

#to get the list of tables or pdfs in the database (lancedb)
def get_existing_tables(db_path):
    db = lancedb.connect(db_path)
    return set(db.table_names())

# <--- Initialization --->
os.makedirs(DATA_DIR, exist_ok=True)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
st.set_page_config(page_title="Hybrid RAG Chatbot", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Hybrid RAG Chatbot</h1>", unsafe_allow_html=True)
st.caption("Ask questions based on your document using Hybrid Search + LLM")

# <---Session state initializations--->
if "db_initialized" not in st.session_state:
    st.session_state["db_initialized"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = [] # for chat display
if "latency_logs" not in st.session_state:
    st.session_state["latency_logs"] = []  # for latency tab
if "retrieval_mode" not in st.session_state:
    st.session_state["retrieval_mode"] = "hybrid" #default = hybrid
if "selected_doc" not in st.session_state:
    st.session_state["selected_doc"] = None
if "prev_selected_doc" not in st.session_state:
    st.session_state["prev_selected_doc"] = None
if "prev_retrieval_mode" not in st.session_state:
    st.session_state["prev_retrieval_mode"] = st.session_state["retrieval_mode"]

# <--- Sidebar: Document Selection/Upload --->
with st.sidebar:
    st.header("📄 Select or Upload Document")
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')] 

    # Only show PDFs that have corresponding LanceDB tables
    existing_tables = get_existing_tables(LANCEDB_PATH)
    valid_docs = [f for f in pdf_files if sanitize_table_name(f) in existing_tables]
    options = valid_docs + ["Upload new document..."]

    selected_doc = st.selectbox("Choose a document:", options, key="selected_doc")

    # Clear chat/logs on doc change
    if (
        st.session_state["prev_selected_doc"] is not None
        and selected_doc != st.session_state["prev_selected_doc"]
        and selected_doc != "Upload new document..."
    ):
        st.session_state["messages"] = []
        st.session_state["latency_logs"] = []

    st.session_state["prev_selected_doc"] = selected_doc

    if selected_doc == "Upload new document...":
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_file:
            file_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success("✅ Document uploaded! It will now be indexed.")
            table_name = sanitize_table_name(uploaded_file.name)
            ingestor = Ingestor(db_path=LANCEDB_PATH, table_name=table_name)
            with st.spinner("🔄 Processing & indexing..."):
                ingestor.run(file_path)
            # Clear chat/logs for new doc
            st.session_state["messages"] = []
            st.session_state["latency_logs"] = []
            st.session_state["db_initialized"] = True
            st.rerun()
    elif selected_doc in valid_docs:
        st.session_state["db_initialized"] = True
    else:
        st.session_state["db_initialized"] = False

# <--- Sidebar: Retrieval Mode --->
with st.sidebar:
    st.header("Retrieval Mode")
    st.selectbox(
        "Choose retrieval mode:",
        options=["hybrid", "dense", "sparse"],
        index=["hybrid", "dense", "sparse"].index(st.session_state["retrieval_mode"]),
        key="retrieval_mode"
    )

    # Clear logs on retrieval mode change
    if st.session_state["retrieval_mode"] != st.session_state["prev_retrieval_mode"]:
        st.session_state["messages"] = []
        st.session_state["latency_logs"] = []
    st.session_state["prev_retrieval_mode"] = st.session_state["retrieval_mode"]


# --- Main Interface: Tabs ---
if (
    st.session_state["db_initialized"]
    and st.session_state["selected_doc"] not in [None, "Upload new document..."]
    and sanitize_table_name(st.session_state["selected_doc"]) in get_existing_tables(LANCEDB_PATH)
):
    table_name = sanitize_table_name(st.session_state["selected_doc"])
    retriever = Retriever(
        db_path=LANCEDB_PATH, table_name=table_name, embedding_model=EMBEDDING_MODEL_NAME
    )
    tab1, tab2 = st.tabs(["💬 Chat", "⏱️ Latency Metrics"])

    with tab1:
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
                results, timings_of_ind_comp = retriever.retrieve(
                    user_input, mode=st.session_state["retrieval_mode"], k=10
                )
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

    with tab2:
        retrieval_mode = st.session_state["retrieval_mode"]
        if retrieval_mode == "hybrid":
            columns = [
                "embedding_time", "dense_search_time", "sparse_search_time",
                "fusion_time", "retrieval_time", "generation_time", "total_time"
            ]
            comp_labels = [
                "Embedding", "Dense Search", "Sparse Search",
                "Fusion", "Retrieval", "Generation", "Total"
            ]
        elif retrieval_mode == "dense":
            columns = [
                "embedding_time", "dense_search_time",
                "retrieval_time", "generation_time", "total_time"
            ]
            comp_labels = [
                "Embedding", "Dense Search",
                "Retrieval", "Generation", "Total"
            ]
        else:  # sparse
            columns = [
                "embedding_time", "sparse_search_time",
                "retrieval_time", "generation_time", "total_time"
            ]
            comp_labels = [
                "Embedding", "Sparse Search",
                "Retrieval", "Generation", "Total"
            ]

        st.header("Latency Metrics")
        if st.session_state["latency_logs"]:
            latest = st.session_state["latency_logs"][-1]
            latest_table = pd.DataFrame(
                {
                    "Components": comp_labels,
                    "Time (s)": [round(latest.get(col, 0), 3) for col in columns],
                }
            )
            st.markdown("**Latest Query**")
            st.table(latest_table)

            df = pd.DataFrame(st.session_state["latency_logs"])
            avg_times = df.mean(numeric_only=True).to_dict()
            avg_table = pd.DataFrame(
                {
                    "Components": comp_labels,
                    "Time (s)": [round(avg_times.get(col, 0), 3) for col in columns],
                }
            )
            st.markdown("**Average Latency (All queries)**")
            st.table(avg_table)

            # CSV data includes queries and responses
            logs_with_queries = df[["query", "response"] + columns].round(3)
            csv = logs_with_queries.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Logs with Responses",
                data=csv,
                file_name="latency_logs_with_responses.csv",
                mime='text/csv'
            )
        else:
            st.info("No latency metrics to display yet. Ask a question!")

else:
    st.info("☝️ Upload a document to begin exploring it using AI!")