from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_pinecone import PineconeRerank

from utils.langchain_setup import (
    get_embeddings,
    get_pinecone_client,
    load_sparse_encoder,
    build_retriever,
    INDEX_NAME,
)

pc = get_pinecone_client()
index = pc.Index(INDEX_NAME)
embeddings = get_embeddings()
sparse_encoder = load_sparse_encoder()
hybrid_retriever_10 = build_retriever(index, embeddings, sparse_encoder, alpha=0.5, top_k=10)

reranker = PineconeRerank(model="bge-reranker-v2-m3", top_n=10)

reranked_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybrid_retriever_10,
)


def show(label: str, docs) -> None:
    print(f"\n{label}")
    for i, doc in enumerate(docs[:5], 1):
        score = doc.metadata.get("relevance_score", doc.metadata.get("score"))
        doc_id = doc.metadata.get("doc_id")
        print(f"  {i}. [{score:.4f}] {doc_id}  {doc.page_content[:70]}")


if __name__ == "__main__":
    query = 'Is it wise to have plenty of current accounts in different banks?'
    print(f"Query: {query}")

    show("Hybrid (alpha=0.5) only", hybrid_retriever_10.invoke(query))
    show("Hybrid + Pinecone rerank (bge-reranker-v2-m3)", reranked_retriever.invoke(query))
