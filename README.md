# Sunbeam Triage Harness

Analyze a Solutions Run UUID and write a self-contained HTML diagnosis:

```bash
uv run sunbeam-triage <uuid>
```

The harness mirrors the full UUID prefix from the configured Swift/RadosGW
container into `artifacts/<uuid>/`, extracts bounded evidence from the run
metadata and Sunbeam logs, asks an OpenRouter-compatible model for a structured
diagnosis, and renders `diagnostics-<uuid>.html`.

## Configuration

Defaults live in `config.toml`. The OpenRouter API key is read from the
environment variable named by `llm.api_key_env`, which defaults to
`OPENROUTER_API_KEY`.

Useful overrides:

```bash
uv run sunbeam-triage <uuid> --model openrouter/auto
uv run sunbeam-triage <uuid> --refresh
uv run sunbeam-triage <uuid> --offline
uv run sunbeam-triage <uuid> --output /tmp/diagnostics-{uuid}.html
```

`--offline` skips Swift downloads and analyzes an already mirrored
`<artifact_root>/<uuid>/` tree.

## Verification

```bash
uv sync --dev
uv run pytest -q
uv run python -m compileall -q analyze.py streamlit_app.py sunbeam_triage
```

Run the Streamlit cockpit with:

```bash
OPENROUTER_API_KEY=<key> uv run streamlit run streamlit_app.py
```

## Multi-model arena

Run the same UUID through several OpenRouter-compatible models and write a
combined comparison report:

```bash
uv run sunbeam-triage-arena run <uuid> --models model/a,model/b
```

If `[arena] models = [...]` is set in `config.toml`, `--models` can be omitted.
Arena records are written under `artifacts/.sunbeam-triage/` as JSON snapshots
plus append-only event logs. The Streamlit cockpit can score completed arenas
with a fixed human rubric; contender model names stay hidden until the verdict is
saved.

Export judged arena records as provider-neutral JSONL:

```bash
uv run sunbeam-triage-arena export --output arena-eval.jsonl
```
