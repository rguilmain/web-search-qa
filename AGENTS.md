# AGENTS.md

Short reference for agents working in this repository.

## Commands

```bash
uv sync                     # install dependencies
uv run web-search-qa        # start the local Gradio UI (127.0.0.1 only)
uv run pytest               # tests (live test skipped unless RUN_LIVE_TESTS=1)
RUN_LIVE_TESTS=1 uv run pytest -m live
uv run ruff check .
uv run ruff format .
```

## Layout

```text
src/web_search_qa/
  config.py       pydantic-settings; env overrides; secret helpers
  llm.py          OpenRouter Chat Completions wrapper + model list
  agent.py        ReAct loop + typed streaming events
  ui.py           Gradio Blocks UI, streaming trace, stop/clear
  tools/
    base.py       Tool, ToolResult, ToolError, registry, helpers
    wikipedia.py  MediaWiki search/page + link extraction
    web_search.py ddgs / Tavily / Brave providers
    fetch.py      HTTP fetch, extraction, SSRF checks, PDF support
    __init__.py   build_registry()
tests/            mocked HTTP + fake LLM tests; test_live.py is opt-in
```

## Conventions

- No agent frameworks. Keep the ReAct loop hand-written and synchronous.
- The agent yields only typed events: thought, tool_call, observation,
  final_answer. Tool errors become observations; never raise them into the
  agent. LLM errors may propagate to the UI.
- Tool callables expose `name`, `description`, `parameters` (JSON Schema), and
  `__call__`, returning `str` or `ToolResult`. Register with `tool_from()`.
- Replay assistant tool-call messages with `reasoning_details`/`reasoning`
  untouched. Keep `tools` in every request that may call tools.
- Validate URLs for public http(s) only, including DNS results and redirects.
  Respect timeouts, size caps, and content-type checks.
- Math output uses `\(...\)` inline and `$$...$$` display. Normalize bare
  `$...$` only in final answers; never disable `sanitize_html`.
- Group a turn's thought/tool/observation events under one ChatMessage parent
  using metadata `id` / `parent_id`; keep the final answer as a top-level message.
- Secrets come only from `.env`/environment via `Settings`. `.env` is
  gitignored; edit `.env.example` when adding a setting.
- Never enable `share=True`, `0.0.0.0`, auth, Docker, or deployment.
- Run `ruff check .` and `pytest` before finishing. `git` was initialized but
  commits are left to the user.
