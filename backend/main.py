"""
FastAPI backend serving Vapi's "Custom LLM" contract.

Endpoints:
- POST /chat/completions  - the endpoint Vapi calls each conversation turn.
                             OpenAI-compatible request/response, SSE streaming.
- GET  /health             - simple liveness check.
- GET  /logs                - recent conversation turns, for the Streamlit
                               dashboard (also usable by any monitoring tool).

Run with:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
from the project root (so the `shared` and `backend` packages resolve).
"""

import json
import time
import uuid
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from shared.config import VAPI_SERVER_SECRET, GROQ_MODEL
from shared.errors import ServiceError
from backend.vapi_models import ChatCompletionRequest
from backend.rag_chain import stream_answer
from backend.call_logger import log_turn, get_recent_turns

app = FastAPI(title="Voice RAG Assistant Backend")

# Allows the Streamlit app (running on a different host/port) to call
# /logs and /chat/completions directly for the dashboard and text-chat
# tester panel. Vapi itself calls server-to-server, so CORS doesn't
# affect that path either way.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _verify_vapi_secret(authorization: Optional[str]) -> None:
    """
    If VAPI_SERVER_SECRET is configured, require it as a Bearer token on
    every request, so random requests to your public URL can't consume
    your Jina/Pinecone/Groq quota. Configure the same value on Vapi's side
    via the /credential endpoint (provider: "custom-llm", apiKey: <secret>).
    If VAPI_SERVER_SECRET is left empty, auth is skipped (fine for local
    testing, not recommended once deployed publicly).
    """
    if not VAPI_SERVER_SECRET:
        return
    expected = f"Bearer {VAPI_SERVER_SECRET}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing server secret.")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/logs")
async def logs(limit: int = 50):
    return {"turns": get_recent_turns(limit=limit)}


@app.post("/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(default=None)):
    _verify_vapi_secret(authorization)

    try:
        body = await request.json()
        payload = ChatCompletionRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Malformed request: {e}")

    if not payload.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty.")

    # The last message is the user's current turn; everything before it is
    # conversation history (used for follow-up-question context).
    latest = payload.messages[-1]
    history = [{"role": m.role, "content": m.content} for m in payload.messages[:-1]]
    question = latest.content

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model_name = payload.model or GROQ_MODEL

    if payload.stream:
        return StreamingResponse(
            _sse_stream(question, history, completion_id, created, model_name),
            media_type="text/event-stream",
        )
    else:
        return await _full_completion(question, history, completion_id, created, model_name)


async def _sse_stream(question: str, history: list, completion_id: str, created: int, model_name: str):
    t0 = time.perf_counter()
    full_answer = []
    error_message = None

    def _chunk(delta: dict, finish_reason=None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    try:
        async for token in stream_answer(question, history):
            full_answer.append(token)
            yield _chunk({"content": token})
    except ServiceError as e:
        error_message = f"{e.service} error: {e.friendly_message}"
        # Speak the error rather than leaving the caller in silence.
        yield _chunk({"content": f"Sorry, I ran into a problem: {e.friendly_message}"})
    except Exception as e:
        error_message = f"Unexpected error: {e}"
        yield _chunk({"content": "Sorry, I ran into an unexpected problem answering that."})

    yield _chunk({}, finish_reason="stop")
    yield "data: [DONE]\n\n"

    log_turn(
        {
            "question": question,
            "answer": "".join(full_answer) if not error_message else None,
            "error": error_message,
            "total_latency_s": round(time.perf_counter() - t0, 3),
        }
    )


async def _full_completion(question: str, history: list, completion_id: str, created: int, model_name: str):
    t0 = time.perf_counter()
    full_answer = []
    error_message = None

    try:
        async for token in stream_answer(question, history):
            full_answer.append(token)
    except ServiceError as e:
        error_message = f"{e.service} error: {e.friendly_message}"
    except Exception as e:
        error_message = f"Unexpected error: {e}"

    answer_text = "".join(full_answer) if not error_message else f"Sorry, I ran into a problem: {error_message}"

    log_turn(
        {
            "question": question,
            "answer": "".join(full_answer) if not error_message else None,
            "error": error_message,
            "total_latency_s": round(time.perf_counter() - t0, 3),
        }
    )

    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer_text},
                    "finish_reason": "stop",
                }
            ],
        }
    )
