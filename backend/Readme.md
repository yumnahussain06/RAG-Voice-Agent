# Voice RAG Assistant (Vapi + LangChain + Pinecone + Jina + Groq)

A real-time, voice-first RAG chatbot: Vapi handles speech-to-text and
text-to-speech over a phone call or browser call, while your own backend
runs a LangChain retrieval-augmented pipeline to generate grounded,
knowledge-based answers that Vapi speaks back to the caller as they're
generated.

This is a companion project to the text-only RAG assistant built earlier
in this conversation - same underlying ideas (Pinecone vector store, Jina
embeddings, Groq generation, no local DB/embedders), but restructured
around LangChain and Vapi's real-time voice contract, with its own
separate Pinecone knowledge base.

## Stack

| Layer | Service |
|---|---|
| Voice (STT + TTS + call handling) | Vapi |
| Backend framework | FastAPI (async, required for streaming) |
| Orchestration | LangChain (`langchain-core`, `langchain-groq`) |
| Vector store | Pinecone (serverless), separate index from the text-RAG project |
| Embeddings | Jina AI (`jina-embeddings-v3`), wrapped as a LangChain `Embeddings` class |
| Generation LLM | Groq, via `langchain-groq`'s `ChatGroq` |
| Admin/testing UI | Streamlit (document management, text-chat tester, call log dashboard) |

## How Vapi actually connects to this backend

Vapi's **Custom LLM** integration is what this project uses. Concretely:

1. A caller speaks; Vapi converts speech to text.
2. Vapi sends a POST request to your backend's `/chat/completions`,
   formatted like an OpenAI chat completion request
   (`model`, `messages`, `stream: true`).
3. Your backend is responsible for generating the reply - this is where
   the RAG pipeline runs (retrieve from Pinecone, generate with Groq).
4. Your backend streams the reply back as Server-Sent Events, in OpenAI's
   `chat.completion.chunk` format.
5. Vapi converts the streamed text to speech in real time as it arrives,
   so the caller starts hearing the answer before it's fully generated.

Your backend never calls Vapi directly for the answer - **Vapi calls
you**, once per conversation turn.

## Project structure

```
voice_rag_assistant/
  shared/
    config.py            Settings (.env / st.secrets / env vars)
    errors.py             ServiceError + exception classifiers (Jina/Pinecone/Groq)
    ingestion.py           Extract text (PDF/DOCX/TXT) + chunk with overlap
    jina_embeddings.py      LangChain Embeddings class backed by the Jina API
    vector_store.py          Pinecone wrapper (namespace-per-document)
  backend/
    main.py                    FastAPI app: /chat/completions (Vapi), /health, /logs
    vapi_models.py               Request schema for Vapi's payload
    retriever.py                  Custom LangChain BaseRetriever (multi-namespace Pinecone search)
    rag_chain.py                   Retrieval + prompt + streaming Groq generation
    call_logger.py                  In-memory recent-turns log for the dashboard
  streamlit_app/
    app.py                            Documents / Voice Agent / Test Chat / Call Logs tabs
  requirements-backend.txt
  requirements-streamlit.txt
  .env.example
```

### The Streamlit app's four tabs

- **Documents** - upload/list/delete files in the knowledge base.
- **Voice Agent** - a button that opens the backend's `/voice` page in a
  new browser tab, where the real microphone widget lives (Vapi's
  `html-script-tag` SDK). It's deliberately **not** embedded inside
  Streamlit itself - see "Why the voice widget lives on the backend, not
  in Streamlit" below.
- **Test Chat** - a typed, text-only way to sanity-check the RAG pipeline
  without using your microphone.
- **Call Logs** - recent conversation turns handled by the backend
  (including calls made through the Voice Agent tab or a real phone call).

### Why the voice widget lives on the backend, not in Streamlit

An earlier version of this project embedded Vapi's widget directly inside
the Streamlit app via `st.components.v1.html()`. That doesn't work
reliably: Streamlit renders components inside a sandboxed `srcdoc` iframe
with a `null` origin, and Vapi's calling engine (Daily.co under the hood)
sets up call audio using `postMessage` between frames, which requires a
real page origin. Inside Streamlit's sandbox this fails silently and
loops forever between "connecting..." and disconnecting.

The fix: the widget is served as a real, standalone page directly from
the **backend** (`GET /voice` in `backend/main.py`), with
`VAPI_PUBLIC_KEY`/`VAPI_ASSISTANT_ID` injected server-side into the HTML.
Streamlit's Voice Agent tab just links to it (`st.link_button`), which
opens it as a normal top-level browser tab - a real origin, no
sandboxing, no postMessage failure. This also means `VAPI_PUBLIC_KEY` and
`VAPI_ASSISTANT_ID` only need to be set on the **backend** service now,
not the frontend.

## LangChain concepts used

- **`Embeddings` interface** (`shared/jina_embeddings.py`) - Jina's API
  wrapped to satisfy LangChain's two-method contract
  (`embed_documents`/`embed_query`), so it can plug into LangChain
  components expecting an embeddings object.
- **`BaseRetriever`** (`backend/retriever.py`) - a custom retriever
  implementing `_get_relevant_documents`/`_aget_relevant_documents`,
  wrapping our own multi-namespace Pinecone search (needed because
  `langchain_pinecone`'s built-in retriever only searches one fixed
  namespace per instance, and our documents are split across
  one-namespace-per-file).
- **`ChatGroq`** (`backend/rag_chain.py`) - LangChain's chat model wrapper
  around Groq, used with `streaming=True` and `.astream(messages)` for
  token-by-token generation.
- **Message classes** (`SystemMessage`, `HumanMessage`, `AIMessage`) - used
  to reconstruct the conversation history Vapi sends on each turn, so
  follow-up questions ("what about the second one") still resolve
  correctly.

## Setup

1. Install dependencies (backend and Streamlit app have separate
   requirement files since they're typically deployed separately):
   ```
   pip install -r requirements-backend.txt
   pip install -r requirements-streamlit.txt
   ```

2. Get API keys: Pinecone, Jina AI, Groq (same providers as the text-RAG
   project - see that project's README for signup links).

3. Copy `.env.example` to `.env` in the project root and fill in real
   values.

4. Run the backend (from the project root, so `shared`/`backend` resolve
   as packages):
   ```
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

5. Run the Streamlit admin app (separate terminal):
   ```
   streamlit run streamlit_app/app.py
   ```

6. Index at least one document via the Streamlit "Documents" tab, then
   sanity-check retrieval and generation via the "Test Chat" tab before
   wiring up Vapi.

## Connecting Vapi

Vapi needs a **public** URL to reach your backend - `localhost` won't
work even for local testing.

**For local testing:** expose your local backend with a tunnel (e.g.
`ngrok http 8000`), then use the resulting public URL.

**Configure the assistant in the Vapi dashboard:**
1. Create an assistant, and under its Model settings choose **Custom LLM**.
2. Set the Custom LLM URL to your backend's base URL (e.g.
   `https://your-ngrok-id.ngrok.io` or your deployed backend's URL) -
   Vapi appends `/chat/completions` itself.
3. If you set `VAPI_SERVER_SECRET` in your `.env`, register it as Vapi's
   credential for this assistant (`provider: "custom-llm"`,
   `apiKey: <same value>`) so Vapi authenticates its requests and random
   traffic can't hit your endpoint and burn through your API quota.
4. Set the `model` field to any label - your backend ignores it and
   always uses `GROQ_MODEL` from your config, but the field is required
   by Vapi's schema.
5. Test with a real call (Vapi's dashboard supports test calls without a
   phone number) and confirm answers are grounded in your indexed
   documents.

6. Once you have a public key and assistant ID, set `VAPI_PUBLIC_KEY` and
   `VAPI_ASSISTANT_ID` in your **backend's** `.env`/environment (not the
   frontend's - see "Why the voice widget lives on the backend, not in
   Streamlit" above), then open the Streamlit app's **Voice Agent** tab
   and click through to talk to the assistant directly from your browser,
   without needing an actual phone number or ngrok tunnel for casual
   testing (the backend still needs to be reachable at whatever URL you
   configured Custom LLM with on Vapi's side).

## Error handling

Every external call (Jina, Pinecone, Groq) is wrapped in `shared/errors.py`
into a `ServiceError` with a clear cause (no internet, invalid key, no
credits, rate limited, service down). In the voice path
(`backend/main.py`), a failure doesn't crash the request or leave the
caller in silence - it's spoken back as a short apologetic message
("Sorry, I ran into a problem: ...") and logged with the error reason, so
the call still ends gracefully and the failure is visible in the Call
Logs dashboard.

## Data privacy note

As with the text-RAG project, this architecture sends document content
and conversation text to multiple third-party APIs in plaintext (Vapi,
Jina, Pinecone, Groq) since there's no local inference anywhere. If
you're indexing sensitive documents or expect callers to discuss
sensitive information, review each vendor's data-retention policy before
deploying.

## Notes on call logs and persistence

`backend/call_logger.py` keeps recent turns in memory, scoped to the
backend process's lifetime - it resets on redeploy/restart, and (if you
run multiple backend worker processes) each worker has its own separate
log. This mirrors the "no local DB" deployment constraint from the
text-RAG project. If you need logs to survive restarts or to be shared
across multiple workers, swap this module for a real store (e.g.
Supabase Postgres) - `log_turn()`/`get_recent_turns()` are the only two
functions that would need new implementations.