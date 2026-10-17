"""Ollama (local) and OpenAI-compatible (hosted, opt-in) providers over plain HTTP."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from typing import Any

import httpx

from .base import LLMError


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    try:
        out = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"model did not return valid JSON: {e}") from e
    if not isinstance(out, dict):
        raise LLMError("model returned JSON that is not an object")
    return out


class OllamaLLM:
    """Local model through Ollama. Uses schema-constrained decoding via the `format` parameter."""

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", timeout: float = 300.0,
                 transport: httpx.BaseTransport | None = None) -> None:  # fmt: skip
        from urllib.parse import urlparse

        self.name = f"ollama:{model}"
        self.model, self.base_url = model, base_url.rstrip("/")
        # "local" is a fact about the URL, not about the provider's name: text sent to another machine is remote
        self.is_remote = (urlparse(base_url).hostname or "").lower() not in ("127.0.0.1", "localhost", "::1")
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def complete_json(self, system: str, user: str, schema: dict[str, Any], *, max_tokens: int = 1200) -> dict[str, Any]:
        body = {
            "model": self.model, "stream": False, "format": schema,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 4096},
        }  # fmt: skip
        try:
            r = self._client.post(f"{self.base_url}/api/chat", json=body)
            r.raise_for_status()
            return _parse_json(r.json()["message"]["content"])
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama request failed ({e.__class__.__name__}): is `ollama serve` running and the model pulled?") from e
        except (KeyError, ValueError) as e:
            raise LLMError(f"unexpected Ollama response: {e}") from e


class OpenAICompatLLM:
    """Any OpenAI-compatible /chat/completions endpoint. The key is read from an environment variable."""

    is_remote = True

    def __init__(self, model: str, base_url: str, api_key_env: str = "MEMBRIDGE_LLM_API_KEY", timeout: float = 120.0,
                 transport: httpx.BaseTransport | None = None) -> None:  # fmt: skip
        self.name = f"openai-compat:{model}"
        self.model, self.base_url, self.api_key_env = model, base_url.rstrip("/"), api_key_env
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def complete_json(self, system: str, user: str, schema: dict[str, Any], *, max_tokens: int = 1200) -> dict[str, Any]:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise LLMError(f"environment variable {self.api_key_env} is not set")
        body = {
            "model": self.model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "result", "schema": schema, "strict": False}},
        }  # fmt: skip
        try:
            r = self._client.post(f"{self.base_url}/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            return _parse_json(r.json()["choices"][0]["message"]["content"])
        except httpx.HTTPError as e:
            raise LLMError(f"hosted LLM request failed ({e.__class__.__name__})") from e
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"unexpected hosted LLM response: {e}") from e


_NO_TOOLS = "Bash,Edit,Write,Read,WebFetch,WebSearch,Glob,Grep,Task,NotebookEdit,TodoWrite"


class ClaudeCLILLM:
    """Hosted extraction through the user's own logged-in Claude Code (`claude -p`). Runs with every tool disabled, in an
    empty temp directory (no project files, no CLAUDE.md), with a JSON schema for the output. Remote: GuardedLLM redacts first."""

    is_remote = True

    def __init__(self, model: str = "haiku", timeout: float = 240.0) -> None:
        self.name = f"claude-cli:{model}"
        self.model, self.timeout = model, timeout
        self.cost_usd = 0.0
        self.calls = 0
        self._lock = threading.Lock()

    def complete_json(self, system: str, user: str, schema: dict[str, Any], *, max_tokens: int = 1200) -> dict[str, Any]:
        cmd = ["claude", "-p", "--model", self.model, "--output-format", "json", "--json-schema", json.dumps(schema), "--disallowedTools", _NO_TOOLS]
        try:
            with tempfile.TemporaryDirectory() as empty:
                r = subprocess.run(cmd, input=f"{system}\n\n{user}", capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=self.timeout, cwd=empty)
        except FileNotFoundError as e:
            raise LLMError("the `claude` CLI was not found on PATH") from e
        except subprocess.TimeoutExpired as e:
            raise LLMError("claude CLI timed out") from e
        try:
            d = json.loads(r.stdout)
        except json.JSONDecodeError as e:
            raise LLMError(f"claude CLI returned non-JSON output (exit {r.returncode})") from e
        with self._lock:
            self.calls += 1
            self.cost_usd += float(d.get("total_cost_usd") or 0.0)
        if d.get("is_error"):
            raise LLMError(f"claude CLI error: {str(d.get('result'))[:120]}")
        out = d.get("structured_output")
        if isinstance(out, dict):
            return out
        return _parse_json(str(d.get("result") or ""))
