"""
Request schema for the /chat/completions endpoint Vapi calls.

Vapi's Custom LLM integration sends a payload shaped like an OpenAI chat
completion request. We only need a few of its fields to run our RAG
pipeline; `model_config = {"extra": "ignore"}` means any additional
fields Vapi includes (tools, metadata, call info, etc.) are silently
accepted and ignored instead of causing a validation error - important
since Vapi's exact payload shape can gain fields over time and we don't
want that to break this endpoint.
"""

from typing import List, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    stream: Optional[bool] = True

    model_config = {"extra": "ignore"}
