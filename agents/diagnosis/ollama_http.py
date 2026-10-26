"""The smallest possible Ollama client: plain HTTP, no extra package.

Used for two things: embeddings of the knowledge base (knowledge.py) and the LLM explainer (llm.py).
Both work without Ollama (BM25 only / template explanation), so a failure here is never fatal.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


class OllamaError(RuntimeError):
    pass


def post(path: str, payload: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(f"{HOST}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise OllamaError(f"Ollama not reachable at {HOST}: {e}") from e


def models(timeout: float = 2.0) -> list[str]:
    """Installed models, or [] if Ollama is not running."""
    try:
        with urllib.request.urlopen(f"{HOST}/api/tags", timeout=timeout) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError):
        return []


def chat(model: str, messages: list[dict], json_mode: bool = True, temperature: float = 0.0,
         timeout: float = 180.0) -> str:
    payload = {"model": model, "messages": messages, "stream": False,
               "options": {"temperature": temperature, "num_ctx": 8192}}
    if json_mode:
        payload["format"] = "json"
    return post("/api/chat", payload, timeout)["message"]["content"]


def embed(model: str, texts: list[str], timeout: float = 60.0) -> list[list[float]]:
    return post("/api/embed", {"model": model, "input": texts}, timeout)["embeddings"]
