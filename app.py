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
from preprocess.chunker import chunk_text

DATA_DIR = "rag_chatbot/data"

def sanitize_table_name(name):
    name_no_ext = os.path.splitext(name)[0]
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', name_no_ext)

def get_existing_tables(db_path):
    db = lancedb.connect(db_path)
    return set(db.table_names())

os.makedirs(DATA_DIR, exist_ok=True)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Hybrid RAG Chatbot", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Hybrid RAG Chatbot</h1>", unsafe_allow_html=True)
st.caption("Ask questions based on your document using Hybrid Search + LLM")

# --- Session State Initialization ---
if "db_initialized" not in st.session_state:
    st.session_state["db_initialized"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "latency_logs" not in st.session_state:
    st.session_state["latency_logs"] = []
if "retrieval_mode" not in st.session_state:
    st.session_state["retrieval_mode"] = "hybrid"
if "selected_doc" not in st.session_state:
    st.session_state["selected_doc"] = None
if "prev_selected_doc" not in st.session_state:
    st.session_state["prev_selected_doc"] = None
if "prev_retrieval_mode" not in st.session_state:
    st.session_state["prev_retrieval_mode"] = st.session_state["retrieval_mode"]
if "top_texts" not in st.session_state:
    st.session_state["top_texts"] = []

# --- Sidebar: Upload or Select Document ---
with st.sidebar:
    st.header("📄 Select or Upload Document")
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
    existing_tables = get_existing_tables(LANCEDB_PATH)
    valid_docs = [f for f in pdf_files if sanitize_table_name(f) in existing_tables]
    options = valid_docs + ["Upload new document..."]

    selected_doc = st.selectbox("Choose a document:", options, key="selected_doc")

    if (
        st.session_state["prev_selected_doc"] is not None
        and selected_doc != st.session_state["prev_selected_doc"]
        and selected_doc != "Upload new document..."
    ):
        st.session_state["messages"] = []
        st.session_state["latency_logs"] = []
        st.session_state["top_texts"] = []

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
            st.session_state["messages"] = []
            st.session_state["latency_logs"] = []
            st.session_state["top_texts"] = []
            st.session_state["db_initialized"] = True
            st.rerun()
    elif selected_doc in valid_docs:
        st.session_state["db_initialized"] = True
    else:
        st.session_state["db_initialized"] = False

# --- Sidebar: Retrieval + Hyper Parameters ---
with st.sidebar:
    llm_model = st.selectbox("Select LLM Model", ["Groq", "OpenAI"])
    st.header("Retrieval Mode")
    st.selectbox(
        "Choose retrieval mode:",
        options=["hybrid", "dense", "sparse"],
        index=["hybrid", "dense", "sparse"].index(st.session_state["retrieval_mode"]),
        key="retrieval_mode"
    )

    if st.session_state["retrieval_mode"] != st.session_state["prev_retrieval_mode"]:
        st.session_state["messages"] = []
        st.session_state["latency_logs"] = []
        st.session_state["top_texts"] = []
    st.session_state["prev_retrieval_mode"] = st.session_state["retrieval_mode"]

    st.header("Hyper Parameters")
    retrieval_mode = st.session_state["retrieval_mode"]

    if retrieval_mode == "dense":
        top_k_dense = st.number_input("Top-K Dense", 1, 50, 10, step=1)
        top_k_sparse = None
        rrf_k = top_k_final = None
    elif retrieval_mode == "sparse":
        top_k_sparse = st.number_input("Top-K Sparse", 1, 50, 10, step=1)
        top_k_dense = None
        rrf_k = top_k_final = None
    else:  # hybrid
        top_k_dense = st.number_input("Top-K Dense", 1, 50, 10, step=1)
        top_k_sparse = st.number_input("Top-K Sparse", 1, 50, 10, step=1)
        rrf_k = st.number_input("RRF Fusion Parameter (k)", 10, 60, 60, step=1)
        top_k_final = st.number_input("Final Top-K Results (after fusion)", 1, 20, 5, step=1)

# --- Main Tabs ---
if (
    st.session_state["db_initialized"]
    and st.session_state["selected_doc"] not in [None, "Upload new document..."]
    and sanitize_table_name(st.session_state["selected_doc"]) in get_existing_tables(LANCEDB_PATH)
):
    table_name = sanitize_table_name(st.session_state["selected_doc"])
    retriever = Retriever(
        db_path=LANCEDB_PATH, table_name=table_name, embedding_model=EMBEDDING_MODEL_NAME
    )
    tab1, tab2, tab3 = st.tabs(["Chat Interface", "Retrievel metrics", "Data Evaluation Metrics"])

    with tab1:
        st.subheader("💬 Ask Anything")
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Type your question here...", key="main_chat_input")
        if user_input:
            st.session_state["messages"].append({"role": "user", "content": user_input})

            query_times = {}
            total_start = time.perf_counter()

            with st.spinner("🔍 Retrieving context..."):
                results, timings_of_ind_comp = retriever.retrieve(
                    user_input,
                    mode=st.session_state["retrieval_mode"],
                    k_dense=top_k_dense or 0,
                    k_sparse=top_k_sparse or 0,
                    rrf_k=rrf_k,
                    top_k_final=top_k_final 
                )
            top_texts = list(results["text"])
            st.session_state["top_texts"] = top_texts
            retrieve_end = time.perf_counter()
            query_times["retrieval_time"] = retrieve_end - total_start

            query_times["embedding_time"] = timings_of_ind_comp.get("embedding_time", 0)
            query_times["dense_search_time"] = timings_of_ind_comp.get("dense_search_time", 0)
            query_times["sparse_search_time"] = timings_of_ind_comp.get("sparse_search_time", 0)
            query_times["fusion_time"] = timings_of_ind_comp.get("fusion_time", 0)

            gen_start = time.perf_counter()
            with st.spinner("Generating response..."):
                model_type = "openai" if llm_model == "OpenAI" else "groq"
                answer = get_conversational_answer(top_texts, user_input, model_type)
            gen_end = time.perf_counter()

            query_times["generation_time"] = gen_end - gen_start
            query_times["total_time"] = time.perf_counter() - total_start
            query_times["query"] = user_input
            query_times["response"] = answer
            st.session_state["latency_logs"].append(query_times)
            st.session_state["messages"].append({"role": "assistant", "content": answer})
            st.rerun()

    with tab2:
        st.header("Latency Metrics")
        mode = st.session_state["retrieval_mode"]
        if mode == "hybrid":
            columns = ["embedding_time", "dense_search_time", "sparse_search_time",
                       "fusion_time", "retrieval_time", "generation_time", "total_time"]
            labels = ["Embedding", "Dense", "Sparse", "Fusion", "Retrieval", "Generation", "Total"]
        elif mode == "dense":
            columns = ["embedding_time", "dense_search_time", "retrieval_time", "generation_time", "total_time"]
            labels = ["Embedding", "Dense", "Retrieval", "Generation", "Total"]
        else:
            columns = ["embedding_time", "sparse_search_time", "retrieval_time", "generation_time", "total_time"]
            labels = ["Embedding", "Sparse", "Retrieval", "Generation", "Total"]

        if st.session_state["latency_logs"]:
            latest = st.session_state["latency_logs"][-1]
            st.markdown("**Latest Query**")
            st.table(pd.DataFrame({"Components": labels, "Time (s)": [round(latest.get(c, 0), 3) for c in columns]}))

            df = pd.DataFrame(st.session_state["latency_logs"])
            avg = df.mean(numeric_only=True)
            st.markdown("**Average (All Queries)**")
            st.table(pd.DataFrame({"Components": labels, "Time (s)": [round(avg.get(c, 0), 3) for c in columns]}))

            log_csv = df[["query", "response"] + columns].round(3).to_csv(index=False).encode("utf-8")
            st.download_button(" Download Logs with Responses", data=log_csv,
                               file_name="latency_logs.csv", mime="text/csv")
        else:
            st.info("No latency metrics yet. Ask something first!")

    with tab3:
        st.header("Data Evaluation Metrics")
        db = lancedb.connect("rag_chatbot/data/lancedb")
        table = db.open_table(table_name)
        df = table.to_pandas()
        df["length"] = df["text"].apply(len)

        chunk_size = int(df.get("chunk_size", pd.Series([512])).iloc[0])
        chunk_overlap = int(df.get("chunk_overlap", pd.Series([50])).iloc[0])

        st.subheader("Summary")
        st.markdown(f"- **Total Chunks**: {len(df)}")
        st.markdown(f"- **Chunk Size**: {chunk_size}")
        st.markdown(f"- **Chunk Overlap**: {chunk_overlap}")
        st.markdown(f"- **Average Chunk Length**: {df['length'].mean():.2f} characters")
        st.markdown(f"- **Max Chunk Length**: {df['length'].max()} characters")
        st.markdown(f"- **Min Chunk Length**: {df['length'].min()} characters")

        st.subheader("Sample Chunks")
        st.dataframe(df[["text", "length"]].head(5))

        st.subheader("Download Chunks")
        st.download_button(" Download Chunks as CSV",
                           data=df.assign(text=df["text"].str.replace("\n", " ", regex=False)).to_csv(index=False).encode("utf-8"),
                           file_name=f"{table_name}_chunks.csv",
                           mime="text/csv")

        st.subheader("Chunk Length Distribution")
        st.bar_chart(df["length"])

        if st.session_state["top_texts"]:
            st.subheader(" Retrieved Top Contexts")
            for i, txt in enumerate(st.session_state["top_texts"][:top_k_final]):
                st.markdown(f"**Top {i+1}:** { ' '.join(txt.split()) }")
        else:
            st.info("No retrieved contexts yet. Ask a question in the Chat tab.")
else:
    st.info("☝️ Upload a document to begin exploring it using AI!")