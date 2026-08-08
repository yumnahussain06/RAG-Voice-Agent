"""
Shared error handling for all external API calls (Jina, Pinecone, Groq).

(classify_gemini_exception is kept for parity with the text-RAG project /
in case an evaluation judge is added later, but isn't used by the voice
assistant's core path.)

Every network/API call in this app can fail for reasons the user needs to
act on differently:
- no internet / can't reach the service
- invalid or missing API key
- no credits / quota exhausted
- rate limited (retry later)
- the service itself is down
- something unexpected

ServiceError is a single exception type carrying an already-friendly
message plus which service failed, so app.py can catch one exception type
and show st.error(...) instead of letting a raw traceback crash the app.
"""

import requests


class ServiceError(Exception):
    """A user-facing error from an external service call."""

    def __init__(self, service: str, message: str, cause: Exception = None):
        self.service = service
        self.friendly_message = message
        self.cause = cause
        super().__init__(f"[{service}] {message}")


def classify_requests_exception(service: str, exc: requests.exceptions.RequestException) -> ServiceError:
    """Turn a `requests` exception into a ServiceError with a clear message."""
    if isinstance(exc, requests.exceptions.ConnectionError):
        return ServiceError(
            service,
            f"Could not reach {service}. Check your internet connection and try again.",
            exc,
        )
    if isinstance(exc, requests.exceptions.Timeout):
        return ServiceError(service, f"{service} took too long to respond. Please try again.", exc)

    if isinstance(exc, requests.exceptions.HTTPError):
        status = exc.response.status_code if exc.response is not None else None
        if status in (401, 403):
            return ServiceError(service, f"{service} rejected the API key. Check that it is correct and active.", exc)
        if status == 402:
            return ServiceError(service, f"{service} account has no remaining credits.", exc)
        if status == 429:
            return ServiceError(service, f"{service} rate limit or quota exceeded. Please wait and try again.", exc)
        if status is not None and status >= 500:
            return ServiceError(service, f"{service} is currently unavailable. Please try again shortly.", exc)
        return ServiceError(service, f"{service} returned an error (status {status}).", exc)

    return ServiceError(service, f"Unexpected error while calling {service}: {exc}", exc)


def classify_groq_exception(exc: Exception) -> ServiceError:
    """Turn a groq-sdk exception into a ServiceError with a clear message."""
    import groq

    service = "Groq"
    if isinstance(exc, groq.APIConnectionError):
        return ServiceError(service, "Could not reach Groq. Check your internet connection and try again.", exc)
    if isinstance(exc, groq.APITimeoutError):
        return ServiceError(service, "Groq took too long to respond. Please try again.", exc)
    if isinstance(exc, groq.AuthenticationError):
        return ServiceError(service, "Groq rejected the API key. Check that it is correct and active.", exc)
    if isinstance(exc, groq.RateLimitError):
        return ServiceError(service, "Groq rate limit or quota exceeded. Please wait and try again.", exc)
    if isinstance(exc, groq.PermissionDeniedError):
        return ServiceError(service, "Groq denied access. The account may be out of credits.", exc)
    if isinstance(exc, groq.APIStatusError):
        return ServiceError(service, f"Groq returned an error (status {exc.status_code}).", exc)
    return ServiceError(service, f"Unexpected error while calling Groq: {exc}", exc)


def classify_gemini_exception(exc: Exception) -> ServiceError:
    """Turn a google-genai exception into a ServiceError with a clear message."""
    from google.genai import errors as genai_errors

    service = "Gemini"
    if isinstance(exc, genai_errors.ClientError):
        code = getattr(exc, "code", None)
        if code in (401, 403):
            return ServiceError(service, "Gemini rejected the API key. Check that it is correct and active.", exc)
        if code == 429:
            return ServiceError(service, "Gemini rate limit or quota exceeded. Please wait and try again.", exc)
        return ServiceError(service, f"Gemini returned a client error (code {code}).", exc)
    if isinstance(exc, genai_errors.ServerError):
        return ServiceError(service, "Gemini is currently unavailable. Please try again shortly.", exc)
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return ServiceError(service, "Could not reach Gemini. Check your internet connection and try again.", exc)
    return ServiceError(service, f"Unexpected error while calling Gemini: {exc}", exc)


def classify_pinecone_exception(exc: Exception) -> ServiceError:
    """Turn a pinecone-sdk exception into a ServiceError with a clear message."""
    from pinecone import exceptions as pc_exceptions

    service = "Pinecone"
    if isinstance(exc, pc_exceptions.UnauthorizedException):
        return ServiceError(service, "Pinecone rejected the API key. Check that it is correct and active.", exc)
    if isinstance(exc, pc_exceptions.ForbiddenException):
        return ServiceError(service, "Pinecone denied access. The plan may not allow this action (e.g. index limits).", exc)
    if isinstance(exc, pc_exceptions.NotFoundException):
        return ServiceError(service, "The requested Pinecone resource was not found.", exc)
    if isinstance(exc, pc_exceptions.PineconeApiException):
        status = getattr(exc, "status", None)
        if status == 429:
            return ServiceError(service, "Pinecone rate limit or quota exceeded. Please wait and try again.", exc)
        return ServiceError(service, f"Pinecone returned an error (status {status}).", exc)
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return ServiceError(service, "Could not reach Pinecone. Check your internet connection and try again.", exc)
    return ServiceError(service, f"Unexpected error while calling Pinecone: {exc}", exc)
