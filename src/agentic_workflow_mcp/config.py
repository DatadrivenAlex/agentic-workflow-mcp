"""Environment-backed settings. The default provider is the offline stub."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseModel):
    llm_provider: Literal["stub", "openai"] = "stub"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    memory_dir: Path = Path(".memory")
    log_level: LogLevel = "INFO"
    max_steps: int = Field(default=8, ge=1, le=32)
    short_term_window: int = Field(default=12, ge=1, le=100)

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_level(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @classmethod
    def from_env(cls) -> Settings:
        # An explicit path avoids walking parent directories for a stray .env.
        load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
        return cls.model_validate(
            {
                "llm_provider": os.getenv("LLM_PROVIDER", "stub"),
                "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
                "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "openai_base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                "memory_dir": os.getenv("MEMORY_DIR", ".memory"),
                "log_level": os.getenv("LOG_LEVEL", "INFO"),
                "max_steps": os.getenv("MAX_STEPS", "8"),
                "short_term_window": os.getenv("SHORT_TERM_WINDOW", "12"),
            }
        )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
