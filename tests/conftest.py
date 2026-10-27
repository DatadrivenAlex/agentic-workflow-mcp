import pytest

from agentic_workflow_mcp.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        llm_provider="stub",
        openai_api_key="",
        memory_dir=tmp_path,
        log_level="WARNING",
        max_steps=8,
        short_term_window=12,
    )
