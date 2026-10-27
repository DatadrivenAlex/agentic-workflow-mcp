"""Select the stub or the optional OpenAI client."""

from __future__ import annotations

import logging

from agentic_workflow_mcp.config import Settings
from agentic_workflow_mcp.llm.base import LLM
from agentic_workflow_mcp.llm.openai import OpenAILLM
from agentic_workflow_mcp.llm.stub import StubLLM

logger = logging.getLogger(__name__)


def build_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "openai":
        if settings.openai_api_key.strip():
            return OpenAILLM(
                api_key=settings.openai_api_key.strip(),
                model=settings.openai_model,
                base_url=settings.openai_base_url,
            )
        logger.warning("LLM_PROVIDER=openai but OPENAI_API_KEY is empty; using the offline stub")
    return StubLLM()
