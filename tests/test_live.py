from __future__ import annotations

import pytest

from web_search_qa.agent import Agent, FinalAnswerEvent, ToolCallEvent
from web_search_qa.config import Settings
from web_search_qa.llm import OpenRouterLLM
from web_search_qa.tools import build_registry

pytestmark = pytest.mark.live


def test_live_wikipedia_reference_follow() -> None:
    """End-to-end smoke test: search, read Wikipedia, follow an external reference."""

    settings = Settings()
    if not settings.has_openrouter_key:
        pytest.skip("OPENROUTER_API_KEY is not configured")

    registry = build_registry(settings)
    llm = OpenRouterLLM(settings)
    try:
        agent = Agent(llm, registry, settings)
        events = list(
            agent.run(
                "Open the English Wikipedia article 'Analytical Engine'. Follow one "
                "non-Wikipedia external link or reference from that article, read "
                "it, and answer with one verifiable fact from the external source. "
                "Cite both the Wikipedia page and the external URL.",
                max_steps=8,
            )
        )
    finally:
        llm.close()

    tool_names = [event.name for event in events if isinstance(event, ToolCallEvent)]
    final_answers = [event for event in events if isinstance(event, FinalAnswerEvent)]
    assert "wikipedia_page" in tool_names
    assert "fetch_url" in tool_names
    assert final_answers
    assert "http" in final_answers[-1].text
