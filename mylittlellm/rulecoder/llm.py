"""Minimal client for the local llama.cpp OpenAI-compatible endpoint."""
import json
import os
import time
import urllib.request
from urllib.parse import urlparse

DEFAULT_URL = "http://127.0.0.1:8081/v1"
# Fields of a reply worth keeping next to a generated answer (the text itself is stored separately).
LLM_FIELDS_SAFE = ("finish_reason", "prompt_tokens", "completion_tokens", "prompt_tokens_processed",
                   "prompt_ms", "predicted_ms", "elapsed_seconds")


def chat(messages, *, base_url=None, max_tokens=512, temperature=0, seed=42, timeout=900):
    base_url = base_url or os.environ.get("STUDENT_BASE_URL", DEFAULT_URL)
    if urlparse(base_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The pilot only talks to a loopback student server")
    body = {"model": "student", "messages": messages, "temperature": temperature, "seed": seed,
            "max_tokens": max_tokens, "stream": False, "cache_prompt": True}
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    timings = payload.get("timings", {})
    return {"text": payload["choices"][0]["message"].get("content") or "",
            "finish_reason": payload["choices"][0].get("finish_reason"),
            "prompt_tokens": payload["usage"]["prompt_tokens"],
            "completion_tokens": payload["usage"]["completion_tokens"],
            # prompt_n excludes tokens served from the KV cache, i.e. what the CPU really had to process.
            "prompt_tokens_processed": timings.get("prompt_n"),
            "prompt_ms": timings.get("prompt_ms"), "predicted_ms": timings.get("predicted_ms"),
            "elapsed_seconds": round(time.monotonic() - start, 3)}


def server_model(base_url=None):
    base_url = base_url or os.environ.get("STUDENT_BASE_URL", DEFAULT_URL)
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as response:
        return json.load(response)["data"][0]
