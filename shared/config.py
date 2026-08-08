"""
Shared configuration for both the FastAPI backend and the Streamlit app.

Loads from a .env file first (for local dev and for the FastAPI backend,
which has no concept of st.secrets), then falls back to Streamlit secrets
when running inside Streamlit, then falls back to plain environment
variables (for platforms like Render/Railway that inject env vars
directly rather than via a .env file).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


def _get(key: str, default: str = "") -> str:
    try:
        import streamlit as st

        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


# --- API keys -----------------------------------------------------------
PINECONE_API_KEY = _get("PINECONE_API_KEY")
JINA_API_KEY = _get("JINA_API_KEY")
GROQ_API_KEY = _get("GROQ_API_KEY")

# --- Pinecone -------------------------------------------------------------
# Deliberately a different index name/default from the text-only RAG
# project, so this voice assistant has its own fresh knowledge base.
PINECONE_INDEX_NAME = _get("PINECONE_INDEX_NAME", "voice-rag-knowledge-base")
PINECONE_CLOUD = _get("PINECONE_CLOUD", "aws")
PINECONE_REGION = _get("PINECONE_REGION", "us-east-1")

# --- Embeddings (Jina AI) -------------------------------------------------
JINA_MODEL = _get("JINA_MODEL", "jina-embeddings-v3")
EMBEDDING_DIMENSIONS = int(_get("EMBEDDING_DIMENSIONS", "1024"))

# --- Generation LLM (Groq) -------------------------------------------------
GROQ_MODEL = _get("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- Chunking ---------------------------------------------------------------
CHUNK_SIZE_WORDS = int(_get("CHUNK_SIZE_WORDS", "300"))
CHUNK_OVERLAP_WORDS = int(_get("CHUNK_OVERLAP_WORDS", "30"))

# --- Retrieval ---------------------------------------------------------------
TOP_K = int(_get("TOP_K", "4"))

# --- Vapi / backend security -------------------------------------------------
# Shared secret Vapi sends back to your server so you can verify requests
# actually came from Vapi and not from a random caller of your public URL.
# Configured on the Vapi side via the /credential endpoint (provider:
# "custom-llm", apiKey: this same value).
VAPI_SERVER_SECRET = _get("VAPI_SERVER_SECRET", "")

# --- Backend server settings ---------------------------------------------------
BACKEND_HOST = _get("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(_get("BACKEND_PORT", "8000"))

# URL where the Streamlit app can reach the backend (for the log viewer and
# text-chat tester panel). Defaults to localhost for local dev; set this to
# your deployed backend's public URL in production.
BACKEND_URL = _get("BACKEND_URL", f"http://localhost:{BACKEND_PORT}")

# --- Vapi Web Widget (client-side voice UI) -----------------------------------
# These power the embedded "Voice Agent" tab in the Streamlit app - a real
# microphone widget backed by Vapi, distinct from VAPI_SERVER_SECRET above
# (which authenticates Vapi's server-to-server calls to your backend).
# The public key is safe to expose in client-side code (that's what it's
# for); get it from the Vapi dashboard under API Keys.
VAPI_PUBLIC_KEY = _get("VAPI_PUBLIC_KEY", "")
# The ID of the assistant you configured in the Vapi dashboard to use
# Custom LLM mode pointed at this backend's /chat/completions endpoint.
VAPI_ASSISTANT_ID = _get("VAPI_ASSISTANT_ID", "")