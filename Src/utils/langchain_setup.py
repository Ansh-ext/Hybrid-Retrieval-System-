import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

load_dotenv()

ROOT = Path(__file__).parent.parent.parent
BM25_PARAMS_PATH = ROOT / "indexes" / "bm25_params.json"

INDEX_NAME = "fiqa-hybrid-rag-small"
CLOUD = "aws"
REGION = "us-east-1"
DENSE_DIM = 384  
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    """LangChain's Embeddings interface wrapping all-MiniLM-L6-v2, run locally."""
    return HuggingFaceEmbeddings(
        model_name=DENSE_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # unit-norm -> dotproduct == cosine
    )



def fit_sparse_encoder(corpus_texts: list[str]) -> BM25Encoder:
    """Fit a fresh BM25Encoder on our corpus and persist the params to disk."""
    bm25 = BM25Encoder()
    bm25.fit(corpus_texts)
    BM25_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    bm25.dump(str(BM25_PARAMS_PATH))
    return bm25


def load_sparse_encoder() -> BM25Encoder:
    """Load the BM25Encoder params fit in fit_sparse_encoder() (2_sparse_bm25.py)."""
    if not BM25_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"{BM25_PARAMS_PATH} not found -- run 2_sparse_bm25.py first."
        )
    bm25 = BM25Encoder()
    bm25.load(str(BM25_PARAMS_PATH))
    return bm25


def get_pinecone_client() -> Pinecone:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Set PINECONE_API_KEY in your .env file")
    return Pinecone(api_key=api_key)


def get_or_create_index(pc: Pinecone, index_name: str = INDEX_NAME):
    """dotproduct is mandatory -- it's the only metric that supports sparse-dense vectors."""
    existing = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"Creating index '{index_name}' (dim={DENSE_DIM}, metric=dotproduct)...")
        pc.create_index(
            name=index_name,
            dimension=DENSE_DIM,
            metric="dotproduct",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    else:
        print(f"Index '{index_name}' already exists, reusing it.")
    return pc.Index(index_name)


def build_retriever(
    index,
    embeddings: HuggingFaceEmbeddings,
    sparse_encoder: BM25Encoder,
    alpha: float = 0.5,
    top_k: int = 10,
) -> PineconeHybridSearchRetriever:

    return PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=sparse_encoder,
        index=index,
        alpha=alpha,
        top_k=top_k,
    )
