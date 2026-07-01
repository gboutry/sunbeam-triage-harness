import json

from sunbeam_triage.core.triage_state import (
    BudgetProfile,
    InvestigationState,
    TriageLoopOptions,
    observe_tool_result,
    parse_budget_name,
    resolve_triage_budget,
    tool_observation_confidence,
)


def test_resolve_triage_budget_uses_default_profile():
    options = resolve_triage_budget(BudgetProfile())

    assert options.max_rounds == 12
    assert options.hard_max_rounds == 20
    assert options.stall_limit == 3
    assert options.min_evidence_items == 2


def test_resolve_triage_budget_rejects_rounds_above_hard_limit():
    profile = BudgetProfile(hard_max_rounds=8)

    try:
        resolve_triage_budget(profile, max_tool_rounds=9)
    except ValueError as exc:
        assert "hard max" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_parse_budget_name_rejects_unknown_profile():
    assert parse_budget_name("quick") == "quick"

    try:
        parse_budget_name("surprise")
    except ValueError as exc:
        assert "Unknown triage budget profile" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_investigation_state_stops_after_sufficient_independent_evidence():
    options = TriageLoopOptions(max_rounds=12, hard_max_rounds=20)
    state = InvestigationState(options=options)

    state.apply_observation(
        observe_tool_result(
            "search_artifacts",
            {"pattern": "first error"},
            json.dumps({
                "ok": True,
                "matches": [
                    {
                        "path": "nova-api.log",
                        "line": 1242,
                        "excerpt": "10:42:31 oslo.messaging timeout",
                    }
                ],
            }),
        )
    )
    state.apply_observation(
        observe_tool_result(
            "get_artifact_file",
            {"path": "rabbitmq.log", "line_start": 120, "line_count": 20},
            json.dumps({
                "ok": True,
                "path": "rabbitmq.log",
                "line_start": 120,
                "content": "10:42:29 closing AMQP connection\n",
            }),
        )
    )
    state.alternatives_considered.append({
        "hypothesis": "Database outage",
        "status": "less_likely",
        "reason": "No DB errors near the first failure timestamp.",
    })

    assert state.should_finalize() is True
    assert state.stop_reason == "sufficient_evidence"


def test_investigation_state_requires_targeted_evidence_for_sufficiency():
    options = TriageLoopOptions(max_rounds=12, hard_max_rounds=20)
    state = InvestigationState(options=options)

    state.apply_observation(
        observe_tool_result(
            "search_artifacts",
            {"pattern": "first error"},
            json.dumps({
                "ok": True,
                "matches": [
                    {
                        "path": "nova-api.log",
                        "line": 1242,
                        "excerpt": "10:42:31 oslo.messaging timeout",
                    },
                    {
                        "path": "rabbitmq.log",
                        "line": 120,
                        "excerpt": "10:42:29 closing AMQP connection",
                    },
                ],
            }),
        )
    )
    state.alternatives_considered.append({
        "hypothesis": "Database outage",
        "status": "less_likely",
        "reason": "No DB errors near the first failure timestamp.",
    })

    assert state.should_finalize() is False
    assert state.stop_reason == ""


def test_investigation_state_stalls_after_repeated_non_progress():
    options = TriageLoopOptions(
        max_rounds=12,
        hard_max_rounds=20,
        stall_limit=2,
    )
    state = InvestigationState(options=options)
    observation = observe_tool_result(
        "list_artifact_files",
        {},
        json.dumps({"ok": True, "files": ["generated/sunbeam/output.log"]}),
    )

    state.apply_observation(observation)
    state.apply_observation(observation)

    assert state.should_finalize() is True
    assert state.stop_reason == "stalled"


def test_investigation_state_stops_on_budget_exhaustion_with_partial_summary():
    options = TriageLoopOptions(max_rounds=1, hard_max_rounds=20)
    state = InvestigationState(options=options)
    state.rounds_used = 1

    assert state.should_finalize() is True
    assert state.stop_reason == "budget_exhausted"
    assert "budget_exhausted" in state.to_prompt_summary()


def test_empty_targeted_search_records_missing_evidence_and_allows_finalization():
    options = TriageLoopOptions(max_rounds=12, hard_max_rounds=20)
    state = InvestigationState(options=options)

    state.apply_observation(
        observe_tool_result(
            "search_artifacts",
            {"pattern": "k8s/0.*lost"},
            json.dumps({
                "ok": True,
                "matches": [
                    {
                        "path": "generated/sunbeam/status.txt",
                        "line": 10,
                        "excerpt": "k8s/0 unknown lost agent lost",
                    },
                ],
            }),
        )
    )
    state.apply_observation(
        observe_tool_result(
            "search_sosreport",
            {"archive_path": "sosreport.tar.xz", "pattern": "juju.*unit-k8s"},
            json.dumps({
                "ok": True,
                "matches": [
                    {
                        "path": "var/log/juju/unit-k8s-0.log",
                        "line": 22,
                        "excerpt": "unit-k8s-0 successfully connected",
                    },
                ],
            }),
        )
    )
    state.apply_observation(
        observe_tool_result(
            "search_sosreport",
            {"archive_path": "sosreport.tar.xz", "pattern": "invalid entity|password"},
            json.dumps({"ok": True, "matches": []}),
        )
    )

    assert state.should_finalize() is True
    assert state.stop_reason == "sufficient_evidence"
    assert state.missing_evidence
    assert "invalid entity|password" in state.missing_evidence[0]


def test_truncated_result_counts_as_missing_evidence_and_non_progress():
    options = TriageLoopOptions(max_rounds=12, hard_max_rounds=20, stall_limit=1)
    state = InvestigationState(options=options)

    state.apply_observation(
        observe_tool_result(
            "get_artifact_file",
            {"path": "generated/sunbeam/output.log"},
            json.dumps({
                "ok": False,
                "tool_result_truncated_by_budget": True,
                "error": "Tool result exceeded budget; narrow the request.",
            }),
        )
    )

    assert state.stall_count == 1
    assert state.missing_evidence
    assert state.should_finalize() is True
    assert state.stop_reason == "stalled"


def test_tool_observation_confidence_separates_broad_and_targeted_reads():
    broad = observe_tool_result(
        "search_artifacts",
        {"pattern": "wait timed out"},
        json.dumps({
            "ok": True,
            "matches": [
                {
                    "path": "generated/sunbeam/output.log",
                    "line": 1,
                    "excerpt": "wait timed out",
                }
            ],
        }),
    )
    targeted = observe_tool_result(
        "get_artifact_file",
        {
            "path": "generated/sunbeam/output.log",
            "line_start": 10,
            "line_count": 20,
        },
        json.dumps({
            "ok": True,
            "path": "generated/sunbeam/output.log",
            "line_start": 10,
            "content": "wait timed out\n",
        }),
    )

    assert tool_observation_confidence(broad) == "low"
    assert tool_observation_confidence(targeted) == "high"
