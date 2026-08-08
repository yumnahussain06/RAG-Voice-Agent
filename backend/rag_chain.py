"""
The core RAG pipeline: retrieve relevant chunks for the latest user
message, build a prompt (with conversation history so follow-up questions
like "what about the second one" still make sense), and stream the
answer from Groq via LangChain.

Written as an async generator so the FastAPI endpoint can forward each
token to Vapi as soon as it's produced, rather than waiting for the full
answer - this is what makes the voice response start speaking quickly
instead of the caller sitting in silence during generation.
"""

from typing import AsyncGenerator, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from shared.config import GROQ_API_KEY, GROQ_MODEL
from shared.errors import ServiceError, classify_groq_exception
from shared.jina_embeddings import JinaEmbeddings
from backend.retriever import PineconeMultiNamespaceRetriever

# Voice-specific system prompt. Unlike the text-chat project, this output
# is spoken aloud by Vapi's TTS - so it must never contain markdown,
# bullet points, emoji, or anything that reads awkwardly out loud.
SYSTEM_PROMPT = """You are a helpful voice assistant speaking with the user out loud.
Answer using ONLY the context excerpts provided below from the knowledge base.

Rules:
- This response will be converted to speech, so write in plain spoken
  sentences. Never use markdown, bullet points, numbered lists, headers,
  asterisks, or emoji.
- Keep answers concise and conversational, the way a person would speak
  on a phone call - a few sentences, not a long lecture.
- If the context does not contain enough information to answer, say so
  plainly and offer to help with something else, instead of guessing.
- Use the conversation history to understand follow-up questions, but
  still ground every factual claim in the provided context.

Context excerpts:
{context}
"""

_retriever = None
_llm = None


def _get_retriever() -> PineconeMultiNamespaceRetriever:
    global _retriever
    if _retriever is None:
        _retriever = PineconeMultiNamespaceRetriever(embeddings=JinaEmbeddings())
    return _retriever


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        if not GROQ_API_KEY:
            raise ServiceError("Groq", "No API key configured. Add GROQ_API_KEY to your secrets or .env file.")
        _llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.3, streaming=True)
    return _llm


def _format_context(documents) -> str:
    if not documents:
        return "(no relevant context found)"
    return "\n\n---\n\n".join(doc.page_content for doc in documents)


def _build_messages(question: str, history: List[dict], context: str) -> List[BaseMessage]:
    """
    history: a list of {"role": "user"|"assistant", "content": str} dicts,
    the prior turns of the conversation (excluding the current question),
    as sent to us by Vapi in its OpenAI-style `messages` array.
    """
    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]

    for turn in history:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        # system messages from Vapi (e.g. its own default system prompt)
        # are intentionally skipped - our SYSTEM_PROMPT above replaces it.

    messages.append(HumanMessage(content=question))
    return messages


async def stream_answer(question: str, history: List[dict]) -> AsyncGenerator[str, None]:
    """Yield the answer to `question` piece by piece, as Groq generates it."""
    retriever = _get_retriever()
    llm = _get_llm()

    try:
        documents = await retriever.ainvoke(question)
    except ServiceError:
        raise
    context = _format_context(documents)

    messages = _build_messages(question, history, context)

    try:
        async for chunk in llm.astream(messages):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        # ChatGroq/langchain-groq exceptions wrap the underlying groq-sdk
        # exception; classify_groq_exception inspects the exception itself
        # so it works whether it's a raw groq.GroqError or wrapped by
        # LangChain's own exception types.
        import groq

        if isinstance(e, groq.GroqError):
            raise classify_groq_exception(e) from e
        raise ServiceError("Groq", f"Unexpected error while generating a response: {e}", e) from e
