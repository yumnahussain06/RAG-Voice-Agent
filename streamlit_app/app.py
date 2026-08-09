"""
Streamlit companion app for the voice RAG assistant.

- Documents tab: upload/list/delete files in the same Pinecone knowledge
  base the voice assistant (backend) retrieves from.
- Test Chat tab: a text-only way to sanity-check the RAG pipeline without
  needing an actual phone call through Vapi - hits the same
  /chat/completions endpoint Vapi calls, non-streaming for simplicity.
- Call Logs tab: recent conversation turns handled by the backend
  (including real voice calls from Vapi), fetched from the backend's
  /logs endpoint.

This app does not talk to Pinecone/Jina/Groq for the chat feature - it
only talks to the backend's HTTP API for that, so it accurately reflects
what the deployed voice assistant actually does. Document management
talks to Pinecone/Jina directly, same as the text-RAG project's sidebar,
since indexing doesn't need to go through the backend.
"""

import sys
from pathlib import Path

# Make the `shared` package importable regardless of the working directory
# Streamlit was launched from.
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import streamlit as st

from shared.ingestion import extract_text, chunk_text
from shared.jina_embeddings import JinaEmbeddings
from shared.vector_store import upsert_document, delete_document, list_active_documents
from shared.errors import ServiceError
from shared.config import BACKEND_URL, VAPI_SERVER_SECRET

st.set_page_config(page_title="Voice RAG Assistant - Admin", layout="wide")

_embeddings = JinaEmbeddings()

docs_tab, voice_tab, chat_tab, logs_tab = st.tabs(["Documents", "Voice Agent", "Test Chat", "Call Logs"])


# ---------------------------------------------------------------------------
# Documents tab
# ---------------------------------------------------------------------------
with docs_tab:
    st.title("Knowledge Base")
    st.caption("Documents indexed here are retrievable by the voice assistant in real time.")

    uploaded_files = st.file_uploader(
        "Add documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key="doc_uploader",
    )

    if uploaded_files:
        for f in uploaded_files:
            if st.button(f"Index '{f.name}'", key=f"index_{f.name}"):
                with st.spinner(f"Processing {f.name}..."):
                    try:
                        text = extract_text(f.getvalue(), f.name)
                        chunks = chunk_text(text)
                        if not chunks:
                            st.warning(f"No extractable text found in {f.name}.")
                        else:
                            vectors = _embeddings.embed_documents(chunks)
                            upsert_document(f.name, chunks, vectors)
                            st.success(f"Indexed {f.name} ({len(chunks)} chunks).")
                    except ServiceError as e:
                        st.error(f"{e.service} error: {e.friendly_message}")
                    except Exception as e:
                        st.error(f"Unexpected error while indexing {f.name}: {e}")

    st.divider()
    st.subheader("Active Documents")

    try:
        active_docs = list_active_documents()
    except ServiceError as e:
        active_docs = {}
        st.error(f"Could not reach the vector store ({e.service}): {e.friendly_message}")
    except Exception as e:
        active_docs = {}
        st.error(f"Unexpected error reading the vector store: {e}")

    if not active_docs:
        st.caption("No documents indexed yet.")
    else:
        for filename, chunk_count in active_docs.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(filename)
                st.caption(f"{chunk_count} chunks")
            with col2:
                if st.button("Delete", key=f"delete_{filename}"):
                    try:
                        delete_document(filename)
                        st.rerun()
                    except ServiceError as e:
                        st.error(f"{e.service} error: {e.friendly_message}")
                    except Exception as e:
                        st.error(f"Unexpected error deleting {filename}: {e}")


# ---------------------------------------------------------------------------
# Voice Agent tab - links out to the backend's standalone /voice page
# ---------------------------------------------------------------------------
with voice_tab:
    st.title("Voice Agent")
    st.caption(
        "Vapi's voice call cannot run reliably inside an embedded Streamlit "
        "panel - it needs a real top-level browser page. Click below to open "
        "the voice agent in a new tab instead."
    )

    st.info(
        "Why a new tab: Vapi's calling engine sets up audio using "
        "browser-to-browser messaging that requires a real page address. "
        "Streamlit embeds components inside a sandboxed frame with no "
        "real address, which silently breaks that setup and causes an "
        "endless 'connecting...' loop. Opening a real page avoids this."
    )

    voice_url = f"{BACKEND_URL}/voice"
    st.link_button("Open Voice Agent", voice_url, use_container_width=True)
    st.caption(f"Opens: {voice_url}")


# ---------------------------------------------------------------------------
# Test Chat tab
# ---------------------------------------------------------------------------
with chat_tab:
    st.title("Test Chat")
    st.caption(
        "Sends a request to the backend's /chat/completions endpoint - "
        "the same endpoint Vapi calls during a real voice call - so you "
        "can verify retrieval and answers without placing a call."
    )

    if "test_chat_messages" not in st.session_state:
        st.session_state.test_chat_messages = []

    for msg in st.session_state.test_chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask a question about your documents")

    if user_input:
        st.session_state.test_chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    headers = {}
                    if VAPI_SERVER_SECRET:
                        headers["Authorization"] = f"Bearer {VAPI_SERVER_SECRET}"
                    response = requests.post(
                        f"{BACKEND_URL}/chat/completions",
                        headers=headers,
                        json={
                            "model": "test",
                            "messages": st.session_state.test_chat_messages,
                            "stream": False,
                        },
                        timeout=60,
                    )
                    response.raise_for_status()
                    data = response.json()
                    answer = data["choices"][0]["message"]["content"]
                except requests.exceptions.ConnectionError:
                    answer = f"Could not reach the backend at {BACKEND_URL}. Is it running?"
                except Exception as e:
                    answer = f"Unexpected error calling the backend: {e}"

                st.write(answer)

        st.session_state.test_chat_messages.append({"role": "assistant", "content": answer})


# ---------------------------------------------------------------------------
# Call Logs tab
# ---------------------------------------------------------------------------
with logs_tab:
    st.title("Call Logs")
    st.caption(
        "Recent conversation turns handled by the backend, including real "
        "voice calls from Vapi. Resets when the backend process restarts."
    )

    if st.button("Refresh"):
        st.rerun()

    try:
        response = requests.get(f"{BACKEND_URL}/logs", params={"limit": 100}, timeout=15)
        response.raise_for_status()
        turns = response.json().get("turns", [])
    except requests.exceptions.ConnectionError:
        turns = []
        st.error(f"Could not reach the backend at {BACKEND_URL}. Is it running?")
    except Exception as e:
        turns = []
        st.error(f"Unexpected error fetching logs: {e}")

    if not turns:
        st.caption("No calls logged yet.")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Turns logged", len(turns))
        successful = [t for t in turns if not t.get("error")]
        avg_latency = sum(t["total_latency_s"] for t in successful) / len(successful) if successful else 0
        col2.metric("Avg. latency (s)", f"{avg_latency:.2f}")

        st.divider()
        st.dataframe(turns, use_container_width=True)