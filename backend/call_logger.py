"""
A simple in-memory log of recent conversation turns handled by the
backend, so the Streamlit app can display a "recent calls" dashboard.

This is intentionally in-memory (a bounded deque), not a database - it
resets whenever the backend process restarts. This mirrors the same
"session-only, no local persistence" choice made in the text-RAG project,
just scoped to the backend process's lifetime instead of a browser tab,
since voice calls are handled by the backend rather than the browser.

If you need logs to survive backend restarts/redeploys, swap this for a
managed store (e.g. the same Supabase option discussed for the text-RAG
project) - nothing else in this file's interface would need to change.
"""

import threading
import time
from collections import deque
from typing import Dict, List

_MAX_LOG_ENTRIES = 200
_log: deque = deque(maxlen=_MAX_LOG_ENTRIES)
_lock = threading.Lock()


def log_turn(entry: Dict) -> None:
    entry = {**entry, "timestamp": time.time()}
    with _lock:
        _log.append(entry)


def get_recent_turns(limit: int = 50) -> List[Dict]:
    with _lock:
        items = list(_log)
    return items[-limit:][::-1]  # most recent first
