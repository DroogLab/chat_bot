import os
import streamlit as st

# Try to get API keys from Streamlit secrets first (for cloud deployment)
# Then fall back to environment variables (for local development)
def get_api_key(key_name):
    try:
        # For Streamlit Cloud
        return st.secrets[key_name]
    except:
        # For local development
        return os.getenv(key_name)

GROQ_API_KEY = get_api_key("GROQ_API_KEY")
OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LANCEDB_PATH = "rag_chatbot/data/lancedb"
LANCEDB_TABLE = "documents"

# Validate API keys
def validate_api_keys():
    missing_keys = []
    if not GROQ_API_KEY:
        missing_keys.append("GROQ_API_KEY")
    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")
    
    if missing_keys:
        st.error(f"❌ Missing API keys: {', '.join(missing_keys)}")
        st.info("""
        **For Local Development:**
        - Create a `.env` file in your project root
        - Add your API keys: `GROQ_API_KEY=your_key_here`
        
        **For Streamlit Cloud:**
        - Go to your app settings in Streamlit Cloud
        - Add secrets in the TOML format
        """)
        return False
    return True
