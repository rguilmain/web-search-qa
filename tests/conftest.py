"""Shared test configuration and fixtures."""

from __future__ import annotations

import os

import pytest

from web_search_qa.config import Settings


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("RUN_LIVE_TESTS") == "1":
        return
    skip_live = pytest.mark.skip(
        reason="live test; set RUN_LIVE_TESTS=1 (and OPENROUTER_API_KEY) to run"
    )
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        openrouter_api_key="test-openrouter-key",
        wikipedia_user_agent="web-search-qa-test/0.1 (tests@example.com) bot",
        fetch_user_agent="web-search-qa-test/0.1",
        max_history_turns=2,
        wikipedia_page_chars=12_000,
        fetch_page_chars=12_000,
    )
