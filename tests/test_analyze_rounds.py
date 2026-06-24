import json

from sunbeam_triage.analyze_rounds import main


def test_analyze_rounds_prints_session_summary(tmp_path, capsys):
    path = tmp_path / "session.json"
    path.write_text(
        json.dumps(
            {
                "uuid": "uuid-1",
                "model": "model/a",
                "exchanges": [
                    {
                        "response": {
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "search_artifacts",
                                        "arguments": '{"pattern": "ERROR"}',
                                    },
                                }
                            ],
                            "usage": {"total_tokens": 42, "cost": 0.0042},
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert main([str(path)]) == 0

    output = capsys.readouterr().out
    assert "uuid-1" in output
    assert "exchanges=1" in output
    assert "tool_calls=1" in output
    assert "session_tokens=42" in output
    assert "exchange_tokens=42" in output
    assert "session_cost_usd=0.0042" in output
    assert "exchange_cost_usd=0.0042" in output
