# Sunbeam Triage Harness

Launch the Streamlit cockpit:

```bash
OPENROUTER_API_KEY=<key> uv run sunbeam-triage-ui
```

Analyze a Solutions Run UUID from the command line and write a
self-contained HTML diagnosis:

```bash
uv run sunbeam-triage-cli <uuid>
```

The harness mirrors the full UUID prefix from the configured Swift/RadosGW
container into `artifacts/<uuid>/`, extracts bounded evidence from the run
metadata and Sunbeam logs, asks an OpenRouter-compatible model for a structured
diagnosis, and renders `diagnostics-<uuid>.html`.

`sunbeam-triage <uuid>` is kept as a short alias for the CLI.

## Configuration

Defaults live in `config.toml`. The OpenRouter API key is read from the
environment variable named by `llm.api_key_env`, which defaults to
`OPENROUTER_API_KEY`.

Useful overrides:

```bash
uv run sunbeam-triage-cli <uuid> --model openrouter/auto
uv run sunbeam-triage-cli <uuid> --refresh
uv run sunbeam-triage-cli <uuid> --offline
uv run sunbeam-triage-cli <uuid> --output /tmp/diagnostics-{uuid}.html
```

`--offline` skips Swift downloads and analyzes an already mirrored
`<artifact_root>/<uuid>/` tree.

## Verification

```bash
uv sync --dev
uv run pytest -q
uv run python -m compileall -q sunbeam_triage tests
uv run pre-commit run --all-files
```

Run the Streamlit cockpit with:

```bash
OPENROUTER_API_KEY=<key> uv run sunbeam-triage-ui
```

Additional Streamlit options can be passed through, for example:

```bash
uv run sunbeam-triage-ui --server.port 8502
```

## Multi-model arena

Run the same UUID through several OpenRouter-compatible models and write a
combined comparison report:

```bash
uv run sunbeam-triage-arena run <uuid> --models model/a,model/b
```

If `[arena] models = [...]` is set in `config.toml`, `--models` can be omitted.
Session records are written under `artifacts/.sunbeam-triage/` as JSON
snapshots plus append-only event logs. The Streamlit cockpit can score completed
arenas with a fixed human rubric; contender model names stay hidden until the
verdict is saved. Older diagnosis records under `artifacts/.sunbeam-triage-ui/`
are read for compatibility, but new sessions use `.sunbeam-triage`.

Export judged arena records as provider-neutral JSONL:

```bash
uv run sunbeam-triage-arena export --output arena-eval.jsonl
```
