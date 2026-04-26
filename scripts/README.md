# scripts/

The experiment + analysis pipeline. These are research scripts: they run, but
they assume you bring your own API keys and (for OLMo) your own vLLM server.

## Setup

```bash
pip install openai anthropic numpy matplotlib aiohttp
# Put your keys into a .env at the project root:
#   OPENAI_API_KEY=...
#   ANTHROPIC_API_KEY=...
#   OPENROUTER_API_KEY=...   # optional, for OLMo via OpenRouter
```

All path resolution is relative to the project root (one level above `scripts/`).
`logs/`, `logs_sanitized/`, and `embeddings/` are created on demand and are
gitignored.

## Pipeline

1. **Run sessions.** `gaslight.py` runs one session; `batch_run.py` parallelises
   across runs/seeds/setups. Each session writes a JSON file to
   `logs/<batch>/...run###.json` containing the per-turn user/raw/stored text
   and a `tampered` flag.

2. **Prompted introspection.** `replay.py` re-opens each saved session and asks
   one introspection question. The reply
   plus its parsed `confidence` integer is saved next to the session as
   `*.followup.<tag>.json`.

3. **Sanitize.** (Optional) `sanitize_logs.py` copies `logs/` → `logs_sanitized/`,
   stripping `thinking`, `encrypted_content`, and reasoning blocks. It refuses
   to overwrite anything it can't fully strip.

4. **Classify spontaneous introspection.** `classify_attribution.py` uses
   gpt-5-mini to label each assistant turn as `self` / `system` / `both` /
   `none` based on whether the model is asking "why am I doing this?" or
   blaming the system, etc. Results land under
   `embeddings/attributions/<tag>/`.

5. **Build viewer data.** `build_viewer_data.py` reads only from
   `logs_sanitized/` (with a hard sanitization check) and the gpt-5-mini
   attribution cache, then writes per-session JSONs to `data/` plus a
   `data/manifest.json` index. This is what the static viewer at the repo root
   serves.

6. **Plots / tables.** `make_intro_plot.py`, `make_prompted_plot.py`, and
   `tables.py` produce the figures and stats tables used in the writeup.
   `loader.py` is the single source of truth for batch enumeration; `format_index.py`
   maps a session path to the canonical short index used in the viewer URL.

## Modules

| File | Purpose |
| --- | --- |
| `gaslight.py` | one session of N turns with prepend/prefill/append/insert |
| `batch_run.py` | run K sessions in parallel for one (model, setup, word) cell |
| `chatter.py` | thin provider abstraction (Anthropic / OpenAI / vLLM) |
| `replay.py` | followup/calibration prompts on saved sessions |
| `loader.py` | enumerate `logs/` batches; CellStat / AUC helpers |
| `format_index.py` | canonical short index `<model>-<setup>-<word>-<run>[-tNN]` |
| `sanitize_logs.py` | strip thinking + reasoning from logs |
| `classify_attribution.py` | gpt-5-mini judge of spontaneous introspection |
| `build_viewer_data.py` | emit static JSONs for the viewer |
| `tables.py` | summary stats for the LessWrong post |
| `make_intro_plot.py` | stacked bar — spontaneous introspection by attribution |
| `make_prompted_plot.py` | per-model prompted introspection averaged over words |

## Prompts

- `prompts/v4draft.txt` — calibration prompt ("0–100 probability the visible text of your prior assistant messages contains tokens that did not come from your own output sampling").
- `prompts/classify_attribution.txt` — gpt-5-mini classifier spec for self/system/both/none labels.
