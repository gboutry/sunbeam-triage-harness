# Sunbeam Triage Harness

Analyze a Solutions Run UUID and write a self-contained HTML diagnosis:

```bash
python3 analyze.py <uuid>
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
python3 analyze.py <uuid> --model openrouter/auto
python3 analyze.py <uuid> --refresh
python3 analyze.py <uuid> --offline
python3 analyze.py <uuid> --output /tmp/diagnostics-{uuid}.html
```

`--offline` skips Swift downloads and analyzes an already mirrored
`<artifact_root>/<uuid>/` tree.

## Verification

```bash
python3 -m pytest -q
python3 -m compileall -q analyze.py sunbeam_triage
```
