import lancedb
from embeddings.embedder import embed_chunks
import pandas as pd
import time 

class Retriever:
    def __init__(self, db_path, table_name, embedding_model):
        self.table = lancedb.connect(db_path).open_table(table_name)
        self.embedding_model = embedding_model

    def embed_query(self, query):
        return embed_chunks([query], model_name=self.embedding_model)[0]

    def search_dense(self, query_vector, k=10):
        return self.table.search(query_vector).limit(k).to_pandas()

    def search_sparse(self, query, k=10):
        return self.table.search(query, query_type="fts").limit(k).to_pandas()

    def reciprocal_rank_fusion(self, dense_df, sparse_df, k=60, limit=5):
        vector_ranks = {row['text']: i+1 for i, (_, row) in enumerate(dense_df.iterrows())}
        fts_ranks = {row['text']: i+1 for i, (_, row) in enumerate(sparse_df.iterrows())}
        all_texts = set(vector_ranks) | set(fts_ranks)
        scores = {}
        for text in all_texts:
            score = 0
            if text in vector_ranks: score += 1 / (k + vector_ranks[text])
            if text in fts_ranks: score += 1 / (k + fts_ranks[text])
            scores[text] = score
        sorted_texts = sorted(scores, key=scores.get, reverse=True)[:limit]
        final = []
        for text in sorted_texts:
            src = 'both' if text in vector_ranks and text in fts_ranks else ('vector' if text in vector_ranks else 'fts')
            final.append({'text': text, 'final_score': scores[text], 'search_source': src})
        return pd.DataFrame(final)

    def retrieve(self, query, mode="hybrid", k=10):
        timings = {}

        start = time.perf_counter()
        qvec = self.embed_query(query)
        end = time.perf_counter()
        timings["embedding_time"] = end - start

        start = time.perf_counter()
        dense = self.search_dense(qvec, k=10)
        timings["dense_search_time"] = time.perf_counter() - start

        start = time.perf_counter()
        sparse = self.search_sparse(query, k=10)
        timings["sparse_search_time"] = time.perf_counter() - start

        start = time.perf_counter()
        if mode == "dense":
            results = dense.head(k).copy()
            # Create final_score robustly
            if "score" in results.columns:
                results = results.rename(columns={"score": "final_score"})
            elif "distance" in results.columns:
                results["final_score"] = -results["distance"]  # if distance, negate to make "higher is better"
            else:
                results["final_score"] = None
            results["search_source"] = "vector"
        elif mode == "sparse":
            results = sparse.head(k).copy()
            if "score" in results.columns:
                results = results.rename(columns={"score": "final_score"})
            elif "distance" in results.columns:
                results["final_score"] = -results["distance"]
            else:
                results["final_score"] = None
            results["search_source"] = "fts"
        else:
            results = self.reciprocal_rank_fusion(dense, sparse, k=60, limit=k)
        timings["fusion_time"] = time.perf_counter() - start

        # Always ensure required columns exist
        for col in ["text", "final_score", "search_source"]:
            if col not in results.columns:
                results[col] = None

        return results[["text", "final_score", "search_source"]], timings