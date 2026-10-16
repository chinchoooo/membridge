"""Build the configured provider. Hosted inference needs explicit opt-in (allow_hosted)."""

from __future__ import annotations

import threading
from typing import Any

from ..config import Settings
from .base import LLM, LLMError
from .providers import ClaudeCLILLM, OllamaLLM, OpenAICompatLLM
from .redact import redact


class GuardedLLM:
    """Wraps a provider: redacts before remote calls and records what was sent (counts only, never content)."""

    def __init__(self, inner: LLM, *, redact_remote: bool = True) -> None:
        self.inner, self.redact_remote = inner, redact_remote
        self.name, self.is_remote = inner.name, inner.is_remote
        self.sent_chars = 0
        self.redactions: dict[str, int] = {}
        self._lock = threading.Lock()

    def complete_json(self, system: str, user: str, schema: dict[str, Any], *, max_tokens: int = 1200) -> dict[str, Any]:
        if self.is_remote and self.redact_remote:
            user, counts = redact(user)
            with self._lock:
                for k, v in counts.items():
                    self.redactions[k] = self.redactions.get(k, 0) + v
        if self.is_remote:
            with self._lock:
                self.sent_chars += len(system) + len(user)
        return self.inner.complete_json(system, user, schema, max_tokens=max_tokens)


def build_llm(s: Settings) -> GuardedLLM | None:
    """Return None when no provider is configured: everything still works deterministically."""
    if s.llm_provider == "none":
        return None
    if s.llm_provider == "ollama":
        return GuardedLLM(OllamaLLM(s.llm_model, s.llm_base_url))
    if s.llm_provider == "openai_compat":
        if not s.allow_hosted:
            raise LLMError(
                "Hosted inference sends conversation text to a third party. Set MEMBRIDGE_ALLOW_HOSTED=true to opt in "
                "(text is redacted first unless MEMBRIDGE_REDACT_HOSTED=false)."
            )
        return GuardedLLM(OpenAICompatLLM(s.llm_model, s.llm_base_url, s.llm_api_key_env), redact_remote=s.redact_hosted)
    if s.llm_provider == "claude_cli":
        if not s.allow_hosted:
            raise LLMError("The claude_cli provider sends (redacted) conversation text to Anthropic. Set MEMBRIDGE_ALLOW_HOSTED=true to opt in.")
        return GuardedLLM(ClaudeCLILLM(s.llm_model if s.llm_model not in ("phi3", "") else "haiku"), redact_remote=s.redact_hosted)
    raise LLMError(f"unknown llm_provider '{s.llm_provider}' (none | ollama | openai_compat | claude_cli)")
