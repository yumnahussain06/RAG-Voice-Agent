"""
A LangChain-compatible retriever backed by our own Pinecone
namespace-per-document search (shared.vector_store.query), instead of
langchain_pinecone's built-in VectorStore retriever - which only searches
one fixed namespace per instance, whereas we need to search across every
active document's namespace and merge results by score.

Implementing BaseRetriever's two hook methods lets this object be used
directly inside an LCEL chain (e.g. `retriever | format_docs`), and gives
us both sync and async entry points - the async one matters here since the
FastAPI endpoint that serves Vapi is async and must not block the event
loop on a network call.
"""

import asyncio
from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from shared.jina_embeddings import JinaEmbeddings
from shared import vector_store
from shared.config import TOP_K


class PineconeMultiNamespaceRetriever(BaseRetriever):
    """Retrieves the top-k most relevant chunks across all indexed documents."""

    embeddings: JinaEmbeddings
    top_k: int = TOP_K

    def _matches_to_documents(self, matches: List[dict]) -> List[Document]:
        documents = []
        for m in matches:
            meta = m.get("metadata", {})
            documents.append(
                Document(
                    page_content=meta.get("text", ""),
                    metadata={
                        "filename": meta.get("filename"),
                        "chunk_index": meta.get("chunk_index"),
                        "score": m.get("score"),
                    },
                )
            )
        return documents

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        query_vector = self.embeddings.embed_query(query)
        matches = vector_store.query(query_vector, top_k=self.top_k)
        return self._matches_to_documents(matches)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        # embed_query and vector_store.query are both blocking (network)
        # calls, so run them in a thread to avoid blocking the FastAPI
        # event loop while Vapi is waiting on this request.
        return await asyncio.to_thread(self._get_relevant_documents_sync, query)

    def _get_relevant_documents_sync(self, query: str) -> List[Document]:
        query_vector = self.embeddings.embed_query(query)
        matches = vector_store.query(query_vector, top_k=self.top_k)
        return self._matches_to_documents(matches)
