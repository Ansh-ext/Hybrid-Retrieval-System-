from pathlib import Path
import pandas as pd
from tqdm import tqdm

from utils.langchain_setup import (
    get_embeddings,
    get_or_create_index,
    get_pinecone_client,
    load_sparse_encoder,
    build_retriever,
)

DATA_DIR = Path(__file__).parent.parent / "Notebook"/ "data" / "fiqa"


corpus = pd.read_parquet(DATA_DIR / "corpus_small.parquet")
doc_ids = corpus["_id"].tolist()
doc_texts = corpus["text"].tolist()


embeddings = get_embeddings()
sparse_encoder = load_sparse_encoder()  # fit in 2_sparse_bm25.py

pc = get_pinecone_client()
index = get_or_create_index(pc)

retriever = build_retriever(index, embeddings, sparse_encoder, alpha=0.5, top_k=10)


BATCH = 200
if index.describe_index_stats()["total_vector_count"] < len(doc_ids):
    print(f"Adding {len(doc_ids)} documents through the retriever...")
    for i in tqdm(range(0, len(doc_ids), BATCH), desc="add_texts"):
        batch_ids = doc_ids[i : i + BATCH]
        batch_texts = doc_texts[i : i + BATCH]
        retriever.add_texts(
            texts=batch_texts,
            ids=batch_ids,
            metadatas=[{"doc_id": d} for d in batch_ids],
        )
else:
    print("Index already populated, skipping add_texts.")

if __name__ == "__main__":
    print(index.describe_index_stats())
