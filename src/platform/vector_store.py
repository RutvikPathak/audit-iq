import os
import uuid
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from fastembed import TextEmbedding

QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "audit_documents")

# Initialize Embedding Model
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)

# Initialize Qdrant Client (URL or Embedded Local Storage)
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
    """Split large audit documents into overlapping text chunks."""
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
    """Generate an embedding for a single piece of text."""
    embeddings = list(embedding_model.embed([text]))
    return embeddings[0].tolist()


def _ensure_collection() -> None:
    """Create the Qdrant collection if it does not already exist."""
    collections = qdrant_client.get_collections()
    existing_names = [c.name for c in collections.collections]

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


def index_document(document_id: str, text: str) -> int:
    """Chunk a document, create embeddings, and store chunks in Qdrant."""
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

    return len(chunks)


def search_document(
    document_id: str,
    question: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Search for relevant document chunks using vector similarity."""
    _ensure_collection()
    question_vector = _get_embedding(question)

    query_filter = Filter(
        must=[
            FieldCondition(
                key="document_id",
                match=MatchValue(value=document_id),
            )
        ]
    )

    # Use query_points (qdrant-client >= 1.10.0) with fallback to search
    if hasattr(qdrant_client, "query_points"):
        response = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=question_vector,
            query_filter=query_filter,
            limit=limit,
        )
        points = response.points
        return [
            {
                "text": point.payload.get("text", "") if point.payload else "",
                "score": float(point.score),
                "chunk_index": point.payload.get("chunk_index", 0) if point.payload else 0,
            }
            for point in points
        ]
    else:
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=question_vector,
            query_filter=query_filter,
            limit=limit,
        )
        return [
            {
                "text": result.payload.get("text", "") if result.payload else "",
                "score": float(result.score),
                "chunk_index": result.payload.get("chunk_index", 0) if result.payload else 0,
            }
            for result in results
        ]