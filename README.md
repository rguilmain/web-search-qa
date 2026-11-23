# Web Search QA

A local-only research agent: ask a question in a Gradio chat UI and watch it
search Wikipedia and the web, open pages, follow links, cross-check evidence,
and answer with inline source URLs.

It uses OpenRouter tool calling through the `openai` SDK and a small,
hand-written ReAct loop. There is no agent framework.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- An OpenRouter API key

## Setup

```bash
uv sync
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY=...
```

On Windows PowerShell:

```powershell
uv sync
Copy-Item .env.example .env
notepad .env
```

Everything other than `OPENROUTER_API_KEY` has a working default. `.env` is
gitignored; `.env.example` is committed.

## Run

```bash
uv run web-search-qa
```

The app binds to `127.0.0.1:7860` and opens a browser. It never enables
`share=True`, authentication, Docker, or deployment.

If the key is missing, the UI still starts and shows a warning; research
requests return a clear error until the key is added and the app is restarted.

## Using the UI

- Type a research question and press Enter or **Research**.
- The full research trace — thoughts, tool calls, and observations — streams
  into a single expandable box per turn and collapses when the answer is ready.
  The final answer appears immediately below it as a normal assistant message
  with Markdown links and rendered formulas.
- **Stop** cancels the current run. **Clear** resets the chat and the model's
  conversation history.
- In **Settings** you can choose any OpenRouter model ID (the dropdown is
  populated from OpenRouter's tool-capable model list when reachable),
  enable/disable tools, and change the tool-step limit (default 15). At the
  limit the model is forced to answer from the evidence it has gathered.
- Previous questions and answers are kept as history (last 8 turns by default)
  but tool traces are not replayed as history.

## Math formulas

Formulas render through KaTeX in the chat. Supported delimiters:

- Inline: `\(E=mc^2\)`
- Display: `$$E=mc^2$$` or `\[E=mc^2\]`

The system prompt tells the model to use LaTeX and to avoid backticks around
formulas. If the model writes bare `$E=mc^2$`, the UI safely converts it to
`\(E=mc^2\)` in the final answer only; prices like `$5 and $10` in tool output
are left alone.

For source material, `wikipedia_page` reads the raw article wikitext in the same
API request and includes a “Formulas (LaTeX from article source)” section from
`<math>…</math>` and `{{math|…}}`. `fetch_url` also converts common web math
markup (`<script type="math/tex">`, `<math alttext="…">`, and MathML TeX
annotations) into LaTeX delimiters before extraction.

Limitations: Wikipedia pages that provide only MathML without TeX source, and
scanned/image PDFs, may not yield recoverable LaTeX. You can still ask the
model to restate such formulas in LaTeX from the surrounding context.

## How it works

- `src/web_search_qa/agent.py` runs a bounded ReAct loop and yields typed
  events: `ThoughtEvent`, `ToolCallEvent`, `ObservationEvent`, and
  `FinalAnswerEvent`.
- `src/web_search_qa/llm.py` calls OpenRouter Chat Completions. Assistant tool
  calls are replayed with `reasoning_details`/`reasoning` unchanged, as
  OpenRouter requires for reasoning models.
- `src/web_search_qa/tools/` contains the registry and four tools:
  - `wikipedia_search` / `wikipedia_page` — MediaWiki Action API; page output
    includes clean text plus numbered internal and external/reference links
    with context.
  - `web_search` — keyless `ddgs` by default; optional Tavily/Brave providers.
  - `fetch_url` — main-content extraction with redirects, timeouts, size and
    content-type checks, PDF support, outgoing links, and SSRF blocking of
    non-public addresses.
- Tool failures are returned as observations, never raised into the agent.
  Long observations are truncated with offset guidance; identical calls and
  repeated URL fetches are nudged instead of repeated.
- The system prompt requires tool use for factual questions, link following,
  cross-checking, inline citations, admitting uncertainty, and treating all
  fetched content as untrusted data.

## Search providers

`ddgs` is keyless and requires no configuration. To use a hosted provider:

```dotenv
SEARCH_PROVIDER=tavily
TAVILY_API_KEY=...
```

or:

```dotenv
SEARCH_PROVIDER=brave
BRAVE_API_KEY=...
```

For Wikipedia, `WIKIPEDIA_API_KEY` is optional. If set, it is sent as
`Authorization: Bearer <token>` (Wikimedia OAuth 2.0 access token). Public
reads work without it. Wikimedia requires a descriptive User-Agent with
contact information; `WIKIPEDIA_USER_AGENT` and `FETCH_USER_AGENT` are
configurable and include a sensible default.

## Adding a tool

1. Create a callable object in `src/web_search_qa/tools/` with these
   attributes:

   ```python
   class MyTool:
       name = "my_tool"
       description = "What the model should use this for."
       parameters = {
           "type": "object",
           "properties": {"query": {"type": "string"}},
           "required": ["query"],
       }

       def __call__(self, query: str) -> str | ToolResult: ...
   ```

   Return a string or `ToolResult`. Raise `ToolError` for expected failures;
   the registry converts it to an error observation. Never raise into the
   agent for a recoverable failure.

2. Register it in `build_registry()` in `src/web_search_qa/tools/__init__.py`:

   ```python
   registry.register(tool_from(MyTool()))
   ```

3. Add a test with mocked HTTP in `tests/`. The new tool appears automatically
   in the UI checkbox list and in the model's tool schemas.

## Development

```bash
uv run pytest              # unit tests; live tests are skipped by default
uv run pytest -m live      # runs only live tests (still skipped without RUN_LIVE_TESTS=1)
RUN_LIVE_TESTS=1 uv run pytest -m live   # requires OPENROUTER_API_KEY
uv run ruff check .
uv run ruff format .
```

The live smoke test asks the agent to open a Wikipedia article, follow an
external reference, and cite it. It is skipped unless `RUN_LIVE_TESTS=1` is
set.

## Notes and limitations

- The agent deliberately does not answer factual questions from memory. If
  tools are disabled or unavailable, it will say so.
- Search providers can rate-limit; switch provider or simplify the query.
  `ddgs` manages its own backend selection and timeouts.
- SSRF protection validates hosts and DNS results and checks every redirect,
  but a DNS rebinding attack between validation and connection remains
  theoretically possible in any client that re-resolves the hostname.
- Some pages require JavaScript or block automated access; `fetch_url` reports
  this as an observation instead of failing the run.
- PDF extraction covers text-based PDFs (first 50 pages); scanned PDFs may
  yield no text.
