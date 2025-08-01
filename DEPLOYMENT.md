# Deployment Guide

This guide explains how to configure and deploy your RAG Chatbot both locally and on Streamlit Cloud.

## 🏠 Local Development

### 1. Set up Environment Variables

Create a `.env` file in your project root:

```bash
# Copy the template
cp env_template.txt .env
```

Edit `.env` and add your API keys:

```env
GROQ_API_KEY=your_actual_groq_api_key
OPENAI_API_KEY=your_actual_openai_api_key
```

### 2. Install Dependencies

```bash
# Activate your virtual environment
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run Locally

```bash
streamlit run app.py
```

## ☁️ Streamlit Cloud Deployment

### 1. Push to GitHub

Make sure your code is pushed to a GitHub repository.

### 2. Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Connect your repository
4. Set the main file path to: `app.py`

### 3. Configure Secrets

In your Streamlit Cloud app settings, add the following secrets:

```toml
GROQ_API_KEY = "your_actual_groq_api_key"
OPENAI_API_KEY = "your_actual_openai_api_key"
```

### 4. Deploy

Click "Deploy" and your app will be live!

## 🔧 Configuration Details

### Environment Variables

- **GROQ_API_KEY**: Your Groq API key for LLM responses
- **OPENAI_API_KEY**: Your OpenAI API key for embeddings

### File Structure

```
chat_bot/
├── app.py                 # Main Streamlit app
├── config.py             # Configuration management
├── requirements.txt      # Python dependencies
├── .streamlit/          # Streamlit configuration
│   └── config.toml
├── rag_chatbot/         # Data and embeddings
│   └── data/
│       ├── *.pdf        # Your documents
│       └── lancedb/     # Vector database
└── .env                 # Local environment variables (not in git)
```

## 🚨 Troubleshooting

### Common Issues

1. **Missing API Keys**: The app will show an error message with setup instructions
2. **Import Errors**: Make sure all dependencies are installed
3. **Database Issues**: Ensure LanceDB files are properly indexed

### Local vs Cloud Differences

| Feature | Local | Streamlit Cloud |
|---------|-------|-----------------|
| API Keys | `.env` file | Streamlit secrets |
| File Upload | ✅ Supported | ✅ Supported |
| Database | Local LanceDB | Local LanceDB |
| Environment | Virtual env | Cloud environment |

## 📝 Notes

- The app automatically handles both local and cloud environments
- API keys are validated on startup
- Document processing works the same in both environments
- LanceDB data is included in the repository for cloud deployment 