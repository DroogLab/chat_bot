import lancedb
import pandas as pd
from preprocess.document_loader import load_text
from preprocess.chunker import chunk_text
from embeddings.embedder import embed_chunks

class Ingestor:
    def __init__(self, db_path, table_name):
        self.db = lancedb.connect(db_path)
        self.table_name = table_name

    def run(self, file_path):
        print(f"Loading and chunking {file_path}...")
        text = load_text(file_path)
        chunks = chunk_text(text)
        print("Generating embeddings...")
        dense_vectors = embed_chunks(chunks)
        df = pd.DataFrame({"text": chunks, "vector": list(dense_vectors)})

        if self.table_name in self.db.table_names():
            print("Table exists. Dropping old table...")
            self.db.drop_table(self.table_name)
        table = self.db.create_table(self.table_name, data=df)
        table.create_fts_index("text")
        table.wait_for_index(["text_idx"])
        print("Ingestion and indexing complete.")