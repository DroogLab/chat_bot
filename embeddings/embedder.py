from sentence_transformers import SentenceTransformer

def embed_chunks(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings