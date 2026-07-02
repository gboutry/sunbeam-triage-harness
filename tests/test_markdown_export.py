from sunbeam_triage.core.markdown_export import render_diagnosis_markdown


def test_render_diagnosis_markdown_includes_report_and_chat():
    session = {
        "uuid": "sample-uuid",
        "model": "model/a",
        "updated_at": "2026-07-02T10:00:00Z",
        "failed_step": "sunbeam_deploy",
        "confidence": "supported",
        "triage_confidence": "medium",
        "stop_reason": "sufficient_evidence",
        "summary": "Deploy timed out.",
        "failure_surface": "The wrapper timed out during sunbeam deploy.",
        "root_cause": "Kubernetes readiness did not converge.",
        "evidence": [
            {
                "path": "generated/sunbeam/output.log",
                "line": 42,
                "excerpt": "wait timed out",
            }
        ],
        "recommendations": ["Inspect k8s readiness logs."],
        "unknowns": ["No neutron-server log was inspected."],
        "missing_evidence": ["Need neutron-server timing."],
        "chat": [
            {
                "role": "user",
                "content": "What should I inspect next?",
                "created_at": "2026-07-02T10:01:00Z",
            },
            {
                "role": "assistant",
                "content": "Start with k8s journal logs.",
                "created_at": "2026-07-02T10:01:30Z",
            },
        ],
        "exchanges": [{"request": {"messages": ["raw"]}}],
        "progress_events": [{"message": "raw trace"}],
    }

    markdown = render_diagnosis_markdown(session)

    assert "# Sunbeam triage report: sample-uuid" in markdown
    assert "- UUID: sample-uuid" in markdown
    assert "- Failed step: sunbeam_deploy" in markdown
    assert "- Triage confidence: medium" in markdown
    assert "## Diagnosis" in markdown
    assert "**Summary:** Deploy timed out." in markdown
    assert "**Root cause:** Kubernetes readiness did not converge." in markdown
    assert "## Evidence" in markdown
    assert "- `generated/sunbeam/output.log:42`: wait timed out" in markdown
    assert "## Recommendations" in markdown
    assert "- Inspect k8s readiness logs." in markdown
    assert "## Unknowns" in markdown
    assert "- No neutron-server log was inspected." in markdown
    assert "## Missing Evidence" in markdown
    assert "- Need neutron-server timing." in markdown
    assert "## Conversation" in markdown
    assert "### User - 2026-07-02T10:01:00Z" in markdown
    assert "What should I inspect next?" in markdown
    assert "### Assistant - 2026-07-02T10:01:30Z" in markdown
    assert "Start with k8s journal logs." in markdown
    assert "exchanges" not in markdown
    assert "progress_events" not in markdown
    assert "raw trace" not in markdown


def test_render_diagnosis_markdown_redacts_session_content():
    session = {
        "uuid": "sample-uuid",
        "model": "model/a",
        "summary": "OS_PASSWORD=super-secret-value",
        "root_cause": "Authorization: Bearer sk-or-v1-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "evidence": [
            {
                "path": "generated/sunbeam/output.log",
                "excerpt": "token=AbcdEFGH1234567890abcdEFGH1234567890",
            }
        ],
        "chat": [
            {
                "role": "user",
                "content": "sunbeam enable pro --token secret-token-value",
            }
        ],
    }

    markdown = render_diagnosis_markdown(session)

    assert "super-secret-value" not in markdown
    assert "sk-or-v1-aaaaaaaa" not in markdown
    assert "AbcdEFGH1234567890" not in markdown
    assert "secret-token-value" not in markdown
    assert "OS_PASSWORD=<redacted>" in markdown
    assert "Authorization: Bearer <redacted>" in markdown
    assert "token=<redacted>" in markdown
    assert "sunbeam enable pro --token <redacted>" in markdown


def test_render_diagnosis_markdown_handles_legacy_minimal_session():
    markdown = render_diagnosis_markdown({
        "uuid": "legacy-uuid",
        "summary": "Legacy diagnosis",
        "chat": [],
    })

    assert "# Sunbeam triage report: legacy-uuid" in markdown
    assert "**Summary:** Legacy diagnosis" in markdown
    assert "## Evidence" not in markdown
    assert "## Conversation" in markdown
    assert "No conversation recorded." in markdown
