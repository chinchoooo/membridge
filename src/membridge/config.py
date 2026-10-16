"""Runtime configuration. All values can be overridden with MEMBRIDGE_* env vars."""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEMBRIDGE_", env_file=".env", extra="ignore"
    )

    data_dir: Path = Path.home() / ".membridge"
    import_batch_size: int = 50
    falkor_host: str = "127.0.0.1"
    falkor_port: int = 6390
    falkor_graph: str = "membridge"
    # LLM is optional: with "none" import, graph, retrieval and rule-based extraction all still work.
    llm_provider: str = "none"  # none | ollama | openai_compat
    llm_model: str = "phi3"
    llm_base_url: str = "http://127.0.0.1:11434"  # ollama; for openai_compat e.g. https://api.openai.com/v1
    llm_api_key_env: str = "MEMBRIDGE_LLM_API_KEY"  # NAME of the env var holding the key; never the key itself
    embedder: str = "none"  # none | fastembed (extra `embeddings`) | hashing (lexical fallback, not semantic)
    allow_hosted: bool = False  # explicit opt-in to send (redacted) text to a hosted provider
    redact_hosted: bool = True

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
