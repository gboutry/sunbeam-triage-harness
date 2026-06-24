from sunbeam_triage.tool_activity import analyze_tool_activity


def test_analyze_tool_activity_summarizes_tool_rounds_and_warnings():
    session = {
        "uuid": "uuid-1",
        "model": "model/a",
        "exchanges": [
            {
                "request": {
                    "messages": [
                        {"role": "user", "content": "question"},
                    ],
                    "tools": [{"function": {"name": "get_artifact_file"}}],
                },
                "response": {
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {
                                "name": "get_artifact_file",
                                "arguments": '{"path": "generated/debug.log"}',
                            },
                        }
                    ],
                    "usage": {"total_tokens": 100, "cost": 0.01},
                },
            },
            {
                "request": {
                    "messages": [
                        {"role": "user", "content": "question"},
                        {
                            "role": "tool",
                            "tool_call_id": "call-1",
                            "content": (
                                '{"ok": true, "path": "generated/debug.log", '
                                '"content": "large", '
                                '"tool_result_truncated_by_budget": true}'
                            ),
                        },
                    ],
                },
                "response": {
                    "tool_calls": [
                        {
                            "id": "call-2",
                            "function": {
                                "name": "get_artifact_file",
                                "arguments": '{"path": "generated/debug.log"}',
                            },
                        }
                    ],
                    "usage": {"total_tokens": 200, "cost": 0.02},
                },
            },
            {
                "request": {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "The artifact tool budget is exhausted. "
                                "Answer now."
                            ),
                        }
                    ]
                },
                "response": {"usage": {"total_tokens": 300, "cost": 0.03}},
            },
        ],
    }

    analysis = analyze_tool_activity(session)

    assert analysis["exchange_count"] == 3
    assert analysis["tool_call_count"] == 2
    assert analysis["tool_result_count"] == 1
    assert analysis["tool_result_chars"] > 0
    assert analysis["total_tokens"] == 600
    assert analysis["total_cost"] == 0.06
    assert analysis["repeated_reads"] == [
        {"tool": "get_artifact_file", "target": "generated/debug.log", "count": 2}
    ]
    assert "budget_truncated_tool_result" in analysis["warnings"]
    assert "repeated_read" in analysis["warnings"]
    assert "tool_budget_fallback" in analysis["warnings"]
    assert analysis["rows"][0]["tool_name"] == "get_artifact_file"
    assert analysis["rows"][0]["target"] == "generated/debug.log"
    assert analysis["rows"][0]["result_chars"] > 0
    assert analysis["rows"][0]["cost"] == 0.01


def test_analyze_tool_activity_handles_empty_sessions():
    analysis = analyze_tool_activity({"uuid": "empty"})

    assert analysis["exchange_count"] == 0
    assert analysis["tool_call_count"] == 0
    assert analysis["total_cost"] == 0
    assert analysis["rows"] == []
    assert analysis["warnings"] == []
