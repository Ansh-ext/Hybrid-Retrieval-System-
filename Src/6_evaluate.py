import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_pinecone import PineconeRerank

from utils.langchain_setup import (
    get_embeddings,
    get_pinecone_client,
    load_sparse_encoder,
    build_retriever,
    INDEX_NAME,
)

DATA_DIR = DATA_DIR = Path(__file__).parent.parent / "Notebook"/ "data"
SAMPLE_SIZE = 40
SEED = 42


def ndcg_at_k(predicted_ids: list, relevant: dict, k: int = 10) -> float:
    dcg = sum(
        relevant.get(doc_id, 0) / math.log2(rank + 2)
        for rank, doc_id in enumerate(predicted_ids[:k])
    )
    ideal_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


queries = pd.read_parquet(DATA_DIR / "fiqa" / "queries_small.parquet")
qrels_df = pd.read_parquet(DATA_DIR / "fiqa" / "qrels_small.parquet")

qrels: dict = defaultdict(dict)
for _, row in qrels_df.iterrows():
    qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])

queries_with_qrels = queries[queries["_id"].astype(str).isin(qrels.keys())].copy()
sample = queries_with_qrels.sample(n=SAMPLE_SIZE, random_state=SEED)
print(f"Evaluating on {len(sample)} queries (sampled from {len(queries_with_qrels)})")


pc = get_pinecone_client()
index = pc.Index(INDEX_NAME)
embeddings = get_embeddings()
sparse_encoder = load_sparse_encoder()

sparse_retriever = build_retriever(index, embeddings, sparse_encoder, alpha=0.0, top_k=10)
dense_retriever = build_retriever(index, embeddings, sparse_encoder, alpha=1.0, top_k=10)
hybrid_retriever = build_retriever(index, embeddings, sparse_encoder, alpha=0.5, top_k=10)
hybrid_retriever = build_retriever(index, embeddings, sparse_encoder, alpha=0.7, top_k=10)
hybrid_retriever_10 = build_retriever(index, embeddings, sparse_encoder, alpha=0.5, top_k=10)
hybrid_retriever_20 = build_retriever(index, embeddings, sparse_encoder, alpha=0.5, top_k=20)


reranked_retriever_10 = ContextualCompressionRetriever(
    base_compressor=PineconeRerank(model="bge-reranker-v2-m3", top_n=10),
    base_retriever=hybrid_retriever_10,
)
reranked_retriever_20 = ContextualCompressionRetriever(
    base_compressor=PineconeRerank(model="bge-reranker-v2-m3", top_n=10),
    base_retriever=hybrid_retriever_20,
)


def ids_from(docs) -> list:
    return [d.metadata["doc_id"] for d in docs]



results: dict = defaultdict(list)
for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Evaluating"):
    query_id = str(row["_id"])
    query_text = row["text"]
    relevant = qrels[query_id]

    results["Sparse (BM25)"].append(ndcg_at_k(ids_from(sparse_retriever.invoke(query_text)), relevant))
    results["Dense (all-MiniLM-L6-v2)"].append(ndcg_at_k(ids_from(dense_retriever.invoke(query_text)), relevant))
    results["Hybrid (alpha=0.5)"].append(ndcg_at_k(ids_from(hybrid_retriever.invoke(query_text)), relevant))
    results["Hybrid (alpha=0.7)"].append(ndcg_at_k(ids_from(hybrid_retriever.invoke(query_text)), relevant))
    results["Hybrid + Rerank 10"].append(ndcg_at_k(ids_from(reranked_retriever_10.invoke(query_text)), relevant))
    results["Hybrid + Rerank 20"].append(ndcg_at_k(ids_from(reranked_retriever_20.invoke(query_text)), relevant))

if __name__ == "__main__":
    print(f"\nNDCG@10 x100 on FiQA ({len(sample)} sampled test queries)")
    print("-" * 42)
    for method, scores in results.items():
        print(f"  {method:<26} {np.mean(scores) * 100:.2f}")
