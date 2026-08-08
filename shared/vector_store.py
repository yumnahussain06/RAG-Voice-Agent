"""
Pinecone vector store wrapper - same namespace-per-document design as the
text-only RAG project (see that project's README for the full rationale):
one Pinecone namespace per uploaded document, so add/list/delete are all
simple, atomic operations with no stale chunks left behind.

This module talks to Pinecone directly (not via langchain_pinecone)
because we need multi-namespace search merged by score, which
langchain_pinecone's VectorStore doesn't support out of the box (it
targets one namespace per store instance). shared/retriever.py wraps
`query()` below in a LangChain-compatible BaseRetriever so it still plugs
into an LCEL chain.
"""

from typing import Dict, List

from pinecone import Pinecone, ServerlessSpec
from pinecone import exceptions as pc_exceptions

from shared.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_DIMENSIONS,
    TOP_K,
)
from shared.errors import ServiceError, classify_pinecone_exception

_pc = None
_index = None


def _client() -> Pinecone:
    global _pc
    if _pc is None:
        if not PINECONE_API_KEY:
            raise ServiceError("Pinecone", "No API key configured. Add PINECONE_API_KEY to your secrets or .env file.")
        try:
            _pc = Pinecone(api_key=PINECONE_API_KEY)
        except pc_exceptions.PineconeException as e:
            raise classify_pinecone_exception(e) from e
        except (ConnectionError, TimeoutError) as e:
            raise classify_pinecone_exception(e) from e
    return _pc


def get_index():
    global _index
    if _index is not None:
        return _index

    pc = _client()
    try:
        existing = {idx["name"] for idx in pc.list_indexes()}
        if PINECONE_INDEX_NAME not in existing:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSIONS,
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            )
        _index = pc.Index(PINECONE_INDEX_NAME)
    except pc_exceptions.PineconeException as e:
        raise classify_pinecone_exception(e) from e
    except (ConnectionError, TimeoutError) as e:
        raise classify_pinecone_exception(e) from e

    return _index


def _namespace_for(filename: str) -> str:
    return filename


def upsert_document(filename: str, chunks: List[str], embeddings: List[List[float]]) -> None:
    index = get_index()
    namespace = _namespace_for(filename)
    delete_document(filename)

    vectors = [
        {
            "id": f"{namespace}::chunk-{i}",
            "values": emb,
            "metadata": {"filename": filename, "chunk_index": i, "text": chunk},
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    batch_size = 100
    try:
        for i in range(0, len(vectors), batch_size):
            index.upsert(vectors=vectors[i : i + batch_size], namespace=namespace)
    except pc_exceptions.PineconeException as e:
        raise classify_pinecone_exception(e) from e
    except (ConnectionError, TimeoutError) as e:
        raise classify_pinecone_exception(e) from e


def delete_document(filename: str) -> None:
    index = get_index()
    namespace = _namespace_for(filename)
    try:
        index.delete(delete_all=True, namespace=namespace)
    except pc_exceptions.NotFoundException:
        pass
    except pc_exceptions.PineconeApiException as e:
        if getattr(e, "status", None) == 404:
            pass
        else:
            raise classify_pinecone_exception(e) from e
    except pc_exceptions.PineconeException as e:
        raise classify_pinecone_exception(e) from e
    except (ConnectionError, TimeoutError) as e:
        raise classify_pinecone_exception(e) from e


def list_active_documents() -> Dict[str, int]:
    index = get_index()
    try:
        stats = index.describe_index_stats()
    except pc_exceptions.PineconeException as e:
        raise classify_pinecone_exception(e) from e
    except (ConnectionError, TimeoutError) as e:
        raise classify_pinecone_exception(e) from e

    namespaces = stats.get("namespaces", {}) or {}
    return {ns: info.get("vector_count", 0) for ns, info in namespaces.items() if ns != ""}


def query(query_embedding: List[float], top_k: int = TOP_K) -> List[dict]:
    """Search across every active document namespace, merged by score."""
    index = get_index()
    namespaces = list(list_active_documents().keys())

    all_matches = []
    try:
        for ns in namespaces:
            response = index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=ns,
                include_metadata=True,
            )
            all_matches.extend(response.get("matches", []))
    except pc_exceptions.PineconeException as e:
        raise classify_pinecone_exception(e) from e
    except (ConnectionError, TimeoutError) as e:
        raise classify_pinecone_exception(e) from e

    all_matches.sort(key=lambda m: m["score"], reverse=True)
    return all_matches[:top_k]
