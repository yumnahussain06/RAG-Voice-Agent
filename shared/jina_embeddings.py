"""
Jina AI embeddings, exposed as a LangChain-compatible Embeddings class.

LangChain's Embeddings interface only requires two methods:
    embed_documents(texts: List[str]) -> List[List[float]]
    embed_query(text: str) -> List[float]

Implementing those two lets this class plug directly into LangChain
components (e.g. langchain_pinecone.PineconeVectorStore) that expect an
Embeddings object, while still calling Jina's API directly (no local
model), and still using Jina's asymmetric 'passage' vs 'query' task types
under the hood for better retrieval quality - the same distinction used
in the text-only RAG project's embeddings.py.
"""

from typing import List

import requests
from langchain_core.embeddings import Embeddings

from shared.config import JINA_API_KEY, JINA_MODEL, EMBEDDING_DIMENSIONS
from shared.errors import ServiceError, classify_requests_exception

JINA_EMBEDDINGS_URL = "https://api.jina.ai/v1/embeddings"


def _call_jina(texts: List[str], task: str, dimensions: int) -> List[List[float]]:
    if not JINA_API_KEY:
        raise ServiceError("Jina AI", "No API key configured. Add JINA_API_KEY to your secrets or .env file.")

    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": JINA_MODEL, "task": task, "dimensions": dimensions, "input": texts}

    try:
        response = requests.post(JINA_EMBEDDINGS_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise classify_requests_exception("Jina AI", e) from e
    except ValueError as e:
        raise ServiceError("Jina AI", "Received an unreadable response from Jina AI.", e) from e

    try:
        items = sorted(data["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in items]
    except (KeyError, TypeError) as e:
        raise ServiceError("Jina AI", "Received an unexpected response format from Jina AI.", e) from e


class JinaEmbeddings(Embeddings):
    """LangChain Embeddings implementation backed by the Jina AI API."""

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS):
        self.dimensions = dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document chunks (asymmetric 'passage' task)."""
        return _call_jina(texts, task="retrieval.passage", dimensions=self.dimensions)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single search query (asymmetric 'query' task)."""
        return _call_jina([text], task="retrieval.query", dimensions=self.dimensions)[0]
