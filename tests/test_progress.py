from sunbeam_triage.progress import (
    ProgressEvent,
    event_from_tool_call,
    summarize_progress_events,
)


def test_progress_event_serializes_concise_trace():
    event = ProgressEvent(
        run_id="uuid-1",
        run_type="diagnosis",
        phase="tool_call",
        status="running",
        message="Reading artifact",
        contender_id="A",
        round_number=2,
        tool_name="get_artifact_file",
        target="generated/sunbeam/output.log",
        result_chars=1234,
        total_tokens=456,
        total_cost=0.0123,
        raw={"secret": "not persisted"},
    )

    assert event.to_trace() == {
        "run_id": "uuid-1",
        "run_type": "diagnosis",
        "phase": "tool_call",
        "status": "running",
        "message": "Reading artifact",
        "contender_id": "A",
        "round_number": 2,
        "tool_name": "get_artifact_file",
        "target": "generated/sunbeam/output.log",
        "result_chars": 1234,
        "total_tokens": 456,
        "total_cost": 0.0123,
        "created_at": event.created_at,
    }


def test_event_from_tool_call_extracts_tool_target():
    event = event_from_tool_call(
        {
            "id": "call-1",
            "function": {
                "name": "get_sosreport_file",
                "arguments": (
                    '{"archive_path": "sos.tar.xz", '
                    '"member_path": "var/log/syslog"}'
                ),
            },
        },
        run_id="uuid-1",
        run_type="arena",
        contender_id="B",
        round_number=3,
    )

    assert event.phase == "tool_call"
    assert event.tool_name == "get_sosreport_file"
    assert event.target == "sos.tar.xz::var/log/syslog"
    assert event.message == "Contender B requested get_sosreport_file"


def test_summarize_progress_events_counts_activity():
    summary = summarize_progress_events(
        [
            ProgressEvent(
                run_id="uuid-1",
                run_type="diagnosis",
                phase="model_request",
                status="running",
                message="Querying model",
            ),
            ProgressEvent(
                run_id="uuid-1",
                run_type="diagnosis",
                phase="tool_result",
                status="running",
                message="Tool returned data",
                result_chars=50,
                total_tokens=10,
                total_cost=0.01,
            ),
            ProgressEvent(
                run_id="uuid-1",
                run_type="diagnosis",
                phase="tool_result",
                status="running",
                message="Tool returned more data",
                result_chars=70,
                total_tokens=20,
                total_cost=0.02,
            ),
        ]
    )

    assert summary == {
        "event_count": 3,
        "tool_call_count": 0,
        "tool_result_count": 2,
        "tool_result_chars": 120,
        "total_tokens": 30,
        "total_cost": 0.03,
        "warnings": [],
    }
