from sunbeam_triage.core.redaction import redact_data, redact_text


def test_redact_text_removes_strict_secret_shapes_without_hiding_diagnostics():
    text = "\n".join([
        "generated/sunbeam/output.log:42 failed on 10.1.2.3 host node-a",
        "OS_PASSWORD=super-secret-value",
        "authorization: Bearer sk-or-v1-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "url=https://user:pass@example.invalid/path",
        "jwt=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature",
        "token=AbcdEFGH1234567890abcdEFGH1234567890",
        "-----BEGIN PRIVATE KEY-----",
        "abc123",
        "-----END PRIVATE KEY-----",
    ])

    redacted = redact_text(text)

    assert "super-secret-value" not in redacted
    assert "sk-or-v1-aaaaaaaa" not in redacted
    assert "user:pass@" not in redacted
    assert "eyJhbGciOiJIUzI1NiJ9" not in redacted
    assert "AbcdEFGH1234567890" not in redacted
    assert "abc123" not in redacted
    assert "OS_PASSWORD=<redacted>" in redacted
    assert "authorization: Bearer <redacted>" in redacted
    assert "https://<redacted>@example.invalid/path" in redacted
    assert "<redacted private key block>" in redacted
    assert "generated/sunbeam/output.log:42" in redacted
    assert "10.1.2.3" in redacted
    assert "node-a" in redacted


def test_redact_data_recurses_without_changing_metrics_or_safe_identifiers():
    data = {
        "uuid": "550e8400-e29b-41d4-a716-446655440000",
        "path": "generated/sunbeam/output.log",
        "usage": {"total_tokens": 123, "cost": 0.01},
        "messages": [
            {
                "role": "tool",
                "content": "password: super-secret-value",
            }
        ],
    }

    redacted = redact_data(data)

    assert redacted["uuid"] == "550e8400-e29b-41d4-a716-446655440000"
    assert redacted["path"] == "generated/sunbeam/output.log"
    assert redacted["usage"] == {"total_tokens": 123, "cost": 0.01}
    assert "super-secret-value" not in redacted["messages"][0]["content"]
    assert redacted["messages"][0]["content"] == "password: <redacted>"


def test_redact_text_masks_sunbeam_enable_pro_cli_tokens():
    text = "\n".join([
        "sunbeam enable pro optional-token-1234567890abcdef",
        "sunbeam enable pro --token optional-token-1234567890abcdef --attach",
        "sunbeam enable pro --contract-id contract optional-token-1234567890abcdef",
    ])

    redacted = redact_text(text)

    assert "optional-token-1234567890abcdef" not in redacted
    assert "sunbeam enable pro <redacted>" in redacted
    assert "sunbeam enable pro --token <redacted> --attach" in redacted
    assert "sunbeam enable pro --contract-id contract <redacted>" in redacted


def test_redact_text_ignores_malformed_url_like_log_lines():
    text = "2026-06-23T09:37:00Z \x1b[36;1mproxy= http://squid.internal:3128\x1b[0m"

    redacted = redact_text(text)

    assert "http://squid.internal:3128" in redacted
