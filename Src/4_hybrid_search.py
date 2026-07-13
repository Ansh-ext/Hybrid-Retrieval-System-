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

sparse_retriever = build_retriever(index, embeddings, sparse_encoder, alpha=0.0, top_k=5)
dense_retriever = build_retriever(index, embeddings, sparse_encoder, alpha=1.0, top_k=5)
hybrid_retriever = build_retriever(index, embeddings, sparse_encoder, alpha=0.5, top_k=5)


def show(label: str, docs) -> None:
    print(f"\n{label}")
    for i, doc in enumerate(docs, 1):
        score = doc.metadata.get("score")
        doc_id = doc.metadata.get("doc_id")
        score_str = f"{score:.4f}" if score is not None else "  n/a "
        print(f"  {i}. [{score_str}] {doc_id}  {doc.page_content[:70]}")


if __name__ == "__main__":
    query = 'Is it wise to have plenty of current accounts in different banks?'
    print(f"Query: {query}")

    show("Sparse only (alpha=0.0)", sparse_retriever.invoke(query))
    show("Dense only (alpha=1.0)", dense_retriever.invoke(query))
    show("Hybrid (alpha=0.5)", hybrid_retriever.invoke(query))
