from pathlib import Path
import pandas as pd
from utils.langchain_setup import fit_sparse_encoder, BM25_PARAMS_PATH

DATA_DIR = Path(__file__).parent.parent / "Notebook"/ "data"

corpus = pd.read_parquet(DATA_DIR / "fiqa" / "corpus_small.parquet")
doc_texts = corpus["text"].tolist()

print(f"Fitting BM25Encoder on {len(doc_texts)} documents...")


bm25 = fit_sparse_encoder(doc_texts)
print(f"Saved BM25 params to {BM25_PARAMS_PATH}")



if __name__ == "__main__":
    doc_sparse = bm25.encode_documents(doc_texts[0])
    print(f"\nDoc sparse vector: {len(doc_sparse['indices'])} non-zero terms")

    query = "Where should I park my rainy-day fund?"
    query_sparse = bm25.encode_queries(query)
    print(f"Query: {query!r}")
    print(f"Query sparse vector: {len(query_sparse['indices'])} non-zero terms")
