"""Provider-independent LLM interface. Core storage/retrieval never import a provider."""

from __future__ import annotations

from typing import Any, Protocol


class LLMError(RuntimeError):
    """Provider unavailable, timed out, or returned unusable output."""


class LLM(Protocol):
    name: str
    is_remote: bool  # remote providers get redaction and need explicit opt-in

    def complete_json(self, system: str, user: str, schema: dict[str, Any], *, max_tokens: int = 1200) -> dict[str, Any]:
        """Return a JSON object conforming (as far as the provider can enforce) to `schema`."""
        ...
