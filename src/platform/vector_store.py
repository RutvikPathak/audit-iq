import os
import uuid
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from fastembed import TextEmbedding

QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "audit_documents")

# Initialize Embedding Model
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)

# Initialize Qdrant: Uses URL if Docker is running, otherwise runs locally in ./qdrant_db
if QDRANT_URL and QDRANT_URL.startswith("http"):
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, timeout=3)
    except Exception:
        qdrant_client = QdrantClient(path="./qdrant_db")
else:
    qdrant_client = QdrantClient(path="./qdrant_db")

def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Split large audit documents into overlapping text chunks.
    """

    if not text or not text.strip():
        return []

    text = text.strip()

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - chunk_overlap

    return chunks


def _get_embedding(text: str) -> List[float]:
    """
    Generate an embedding for a single piece of text.
    """

    embeddings = list(
        embedding_model.embed([text])
    )

    return embeddings[0].tolist()


def _ensure_collection() -> None:
    """
    Create the Qdrant collection if it does not already exist.
    """

    collections = qdrant_client.get_collections()

    existing_names = {
        collection.name
        for collection in collections.collections
    }

    if COLLECTION_NAME in existing_names:
        return

    sample_embedding = _get_embedding("AuditIQ initialization")

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=len(sample_embedding),
            distance=Distance.COSINE,
        ),
    )


def index_document(
    document_id: str,
    text: str,
) -> int:
    """
    Chunk a document, create embeddings, and store the chunks in Qdrant.

    Returns:
        Number of chunks indexed.
    """

    if not document_id:
        raise ValueError("document_id is required")

    if not text or not text.strip():
        raise ValueError("Document text cannot be empty")

    _ensure_collection()

    chunks = chunk_text(text)

    if not chunks:
        return 0

    points = []

    for index, chunk in enumerate(chunks):

        vector = _get_embedding(chunk)

        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}_{index}")),
                vector=vector,
                payload={
                    "document_id": document_id,
                    "chunk_index": index,
                    "text": chunk,
                },
            )
        )

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    return len(points)


def search_document(
    document_id: str,
    question: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search Qdrant for the most relevant chunks belonging
    to a specific document.
    """

    if not document_id:
        raise ValueError("document_id is required")

    if not question:
        raise ValueError("question is required")

    _ensure_collection()

    question_vector = _get_embedding(question)

    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=question_vector,
        query_filter={
            "must": [
                {
                    "key": "document_id",
                    "match": {
                        "value": document_id
                    },
                }
            ]
        },
        limit=limit,
    )

    return [
        {
            "text": result.payload.get("text", ""),
            "score": float(result.score),
            "chunk_index": result.payload.get("chunk_index"),
        }
        for result in results
    ]