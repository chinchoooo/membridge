"""Runtime configuration. All values can be overridden with MEMBRIDGE_* env vars."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class Settings(BaseSettings):
    # The optional .env lives in the data directory (~/.membridge/.env), never in the current directory: an IDE can start
    # the MCP server inside any repository, and a `.env` in an untrusted repo must not be able to repoint the graph
    # database or the LLM endpoint at a host that would then receive every memory.
    model_config = SettingsConfigDict(
        env_prefix="MEMBRIDGE_", env_file=Path.home() / ".membridge" / ".env", extra="ignore"
    )

    data_dir: Path = Path.home() / ".membridge"
    import_batch_size: int = 50
    falkor_host: str = "127.0.0.1"
    falkor_port: int = 6390
    falkor_graph: str = "membridge"
    # LLM is optional: with "none" import, graph, retrieval and rule-based extraction all still work.
    llm_provider: str = "none"  # none | ollama | openai_compat | claude_cli
    llm_model: str = "phi3"
    llm_base_url: str = "http://127.0.0.1:11434"  # ollama; for openai_compat e.g. https://api.openai.com/v1
    llm_api_key_env: str = "MEMBRIDGE_LLM_API_KEY"  # NAME of the env var holding the key; never the key itself
    embedder: str = "none"  # none | fastembed (extra `embeddings`) | hashing (lexical fallback, not semantic)
    allow_hosted: bool = False  # explicit opt-in to send (redacted) text to a hosted provider
    redact_hosted: bool = True

    @model_validator(mode="after")
    def _graph_stays_local(self) -> Settings:
        # The projection holds every non-private memory. A remote graph host needs an explicit decision made in the process
        # environment (the container image sets it for its sibling service), not a value any file can supply.
        if self.falkor_host.lower() not in LOOPBACK_HOSTS and os.environ.get("MEMBRIDGE_ALLOW_REMOTE_GRAPH") != "1":
            raise ValueError(
                f"MEMBRIDGE_FALKOR_HOST={self.falkor_host!r} is not a loopback address. To use a remote graph database set "
                "MEMBRIDGE_ALLOW_REMOTE_GRAPH=1 in the process environment."
            )
        return self

    @field_validator("data_dir")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser()

    @property
    def db_path(self) -> Path:
        return self.data_dir / "membridge.db"

    @property
    def import_dir(self) -> Path:
        return self.data_dir / "imports"
