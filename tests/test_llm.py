import io
import json
import tarfile

import pytest

from sunbeam_triage.core.config import Config
from sunbeam_triage.core.llm import DiagnosisReport, OpenRouterClient, REPORT_SCHEMA
from sunbeam_triage.core.triage_state import TriageLoopOptions


class FakeSdkResponse:
    def __init__(self, content, usage=None, tool_calls=None):
        self.choices = [FakeChoice(content, tool_calls=tool_calls)]
        self.usage = usage


class FakeChoice:
    def __init__(self, content, tool_calls=None):
        self.message = FakeMessage(content, tool_calls=tool_calls)


class FakeMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.type = "function"
        self.function = FakeToolFunction(name, arguments)


class FakeToolFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeUsage:
    prompt_tokens = 100
    completion_tokens = 20
    total_tokens = 120
    prompt_tokens_details = {"cached_tokens": 75}
    cache_write_tokens = 25
    cost = 0.00123
    cost_details = {
        "upstream_inference_prompt_cost": 0.0005,
        "upstream_inference_completions_cost": 0.0007,
    }
    is_byok = False


class FakePromptTokensDetails:
    def __init__(self):
        self.cached_tokens = 75


class FakeSdkObjectUsage:
    prompt_tokens = 100
    completion_tokens = 20
    total_tokens = 120
    prompt_tokens_details = FakePromptTokensDetails()


class FakeChat:
    def __init__(self, response):
        if isinstance(response, list):
            self.responses = list(response)
        else:
            self.responses = [response]
        self.calls = []

    def send(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeSdkClient:
    def __init__(self, response):
        self.chat = FakeChat(response)


def _config(model="openrouter/auto"):
    config = Config.load(None)
    config.llm.api_key = "token"
    config.llm.model = model
    return config.llm


def test_openrouter_client_requests_structured_diagnosis_with_sdk():
    response = FakeSdkResponse(
        json.dumps(
            {
                "summary": "The deploy step timed out.",
                "failure_surface": "sunbeam cluster resize timed out.",
                "confidence": "supported",
                "root_cause": "Readiness did not converge before timeout.",
                "evidence": [
                    {
                        "path": "generated/sunbeam/output.log",
                        "line": 2,
                        "excerpt": "wait timed out",
                    }
                ],
                "candidate_mechanisms": [],
                "recommendations": ["Inspect readiness gate."],
                "unknowns": ["No remote CLI completion log found."],
            }
        ),
        usage=FakeUsage(),
    )
    sdk = FakeSdkClient(response)

    report = OpenRouterClient(
        _config(),
        sdk_client=sdk,
    ).diagnose("evidence text", session_id="uuid-diagnosis")

    call = sdk.chat.calls[0]
    assert call["model"] == "openrouter/auto"
    assert call["session_id"] == "uuid-diagnosis"
    assert call["response_format"]["type"] == "json_schema"
    assert call["messages"][1] == {"role": "user", "content": "evidence text"}
    assert report.summary == "The deploy step timed out."
    assert report.evidence[0].line == 2


def test_openrouter_client_includes_response_preview_for_invalid_json():
    sdk = FakeSdkClient(FakeSdkResponse("not json from model"))

    with pytest.raises(RuntimeError, match="not json from model"):
        OpenRouterClient(_config(), sdk_client=sdk).diagnose("evidence text")


def test_openrouter_client_includes_response_preview_for_non_object_json():
    sdk = FakeSdkClient(FakeSdkResponse(json.dumps(["tool required", "no object"])))

    with pytest.raises(RuntimeError, match=r'\["tool required", "no object"\]'):
        OpenRouterClient(_config(), sdk_client=sdk).diagnose("evidence text")


def test_diagnosis_report_defaults_needs_more_evidence_for_old_payloads():
    report = DiagnosisReport.from_dict(
        {
            "summary": "summary",
            "failure_surface": "surface",
            "confidence": "supported",
            "root_cause": "cause",
        }
    )

    assert report.needs_more_evidence is False


def test_diagnosis_report_reads_optional_triage_v2_fields():
    report = DiagnosisReport.from_dict(
        {
            "summary": "summary",
            "failure_surface": "surface",
            "confidence": "supported",
            "root_cause": "cause",
            "triage_confidence": "medium",
            "failure_timeline": [
                {
                    "timestamp": "10:42:29",
                    "source": "rabbitmq.log",
                    "location": "line 120",
                    "event": "RabbitMQ closed the connection.",
                }
            ],
            "cascading_errors": [
                {
                    "path": "nova-api.log",
                    "line": 1242,
                    "excerpt": "oslo.messaging timeout",
                }
            ],
            "alternatives_considered": [
                {
                    "hypothesis": "Database outage",
                    "status": "less_likely",
                    "reason": "No DB errors near the first failure timestamp.",
                }
            ],
            "missing_evidence": ["Need neutron-server timing."],
            "stop_reason": "sufficient_evidence",
        }
    )

    assert report.triage_confidence == "medium"
    assert report.failure_timeline[0].source == "rabbitmq.log"
    assert report.cascading_errors[0].path == "nova-api.log"
    assert report.alternatives_considered[0].hypothesis == "Database outage"
    assert report.missing_evidence == ["Need neutron-server timing."]
    assert report.stop_reason == "sufficient_evidence"


def test_report_schema_requires_needs_more_evidence():
    assert "needs_more_evidence" in REPORT_SCHEMA["schema"]["required"]
    assert REPORT_SCHEMA["schema"]["properties"]["needs_more_evidence"] == {
        "type": "boolean"
    }
    for field in (
        "triage_confidence",
        "failure_timeline",
        "cascading_errors",
        "alternatives_considered",
        "missing_evidence",
        "stop_reason",
    ):
        assert field in REPORT_SCHEMA["schema"]["required"]


def test_openrouter_client_sends_follow_up_chat_with_sdk_session_id():
    sdk = FakeSdkClient(FakeSdkResponse("Inspect the readiness gate."))
    client = OpenRouterClient(_config(), sdk_client=sdk)

    answer = client.chat(
        "diagnosis context",
        [{"role": "user", "content": "What should I inspect next?"}],
        session_id="uuid-chat",
    )

    call = sdk.chat.calls[0]
    assert answer == "Inspect the readiness gate."
    assert call["model"] == "openrouter/auto"
    assert call["session_id"] == "uuid-chat"
    assert "response_format" not in call
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1] == {
        "role": "user",
        "content": "diagnosis context",
    }
    assert call["messages"][2] == {
        "role": "user",
        "content": "What should I inspect next?",
    }


def test_openrouter_client_adds_cache_control_for_anthropic_models():
    sdk = FakeSdkClient(FakeSdkResponse("ok"))
    client = OpenRouterClient(_config("anthropic/claude-sonnet-4"), sdk_client=sdk)

    client.chat("context", [], session_id="uuid-chat")

    assert sdk.chat.calls[0]["cache_control"] == {"type": "ephemeral"}


def test_openrouter_client_does_not_force_cache_control_for_other_models():
    sdk = FakeSdkClient(FakeSdkResponse("ok"))
    client = OpenRouterClient(_config("deepseek/deepseek-v4-pro:floor"), sdk_client=sdk)

    client.chat("context", [], session_id="uuid-chat")

    assert "cache_control" not in sdk.chat.calls[0]


def test_openrouter_client_records_redacted_exchanges_and_usage_metrics():
    sdk = FakeSdkClient(FakeSdkResponse("ok", usage=FakeUsage()))
    client = OpenRouterClient(_config("openrouter/auto"), sdk_client=sdk)

    client.chat("context", [], session_id="uuid-chat")

    assert client.exchanges == [
        {
            "request": {
                "model": "openrouter/auto",
                "messages": sdk.chat.calls[0]["messages"],
                "session_id": "uuid-chat",
            },
            "response": {
                "content": "ok",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "prompt_tokens_details": {"cached_tokens": 75},
                    "cache_write_tokens": 25,
                    "cost": 0.00123,
                    "cost_details": {
                        "upstream_inference_prompt_cost": 0.0005,
                        "upstream_inference_completions_cost": 0.0007,
                    },
                    "is_byok": False,
                },
            },
        }
    ]


def test_openrouter_client_records_json_serializable_sdk_usage_objects():
    sdk = FakeSdkClient(FakeSdkResponse("ok", usage=FakeSdkObjectUsage()))
    client = OpenRouterClient(_config("openrouter/auto"), sdk_client=sdk)

    client.chat("context", [], session_id="uuid-chat")

    json.dumps(client.exchanges)
    assert client.exchanges[0]["response"]["usage"]["prompt_tokens_details"] == {
        "cached_tokens": 75
    }


def test_openrouter_client_executes_artifact_tool_calls_for_diagnosis(tmp_path):
    artifact_root = tmp_path / "uuid"
    output = artifact_root / "generated/sunbeam/output.log"
    output.parent.mkdir(parents=True)
    output.write_text("wait timed out\nretry failed", encoding="utf-8")
    sdk = FakeSdkClient(
        [
            FakeSdkResponse(
                "",
                tool_calls=[
                    FakeToolCall("call-1", "list_artifact_files", "{}"),
                    FakeToolCall(
                        "call-2",
                        "get_artifact_file",
                        json.dumps({"path": "generated/sunbeam/output.log"}),
                    ),
                ],
            ),
            FakeSdkResponse(
                json.dumps(
                    {
                        "summary": "The deploy step timed out.",
                        "failure_surface": "sunbeam cluster resize timed out.",
                        "confidence": "supported",
                        "root_cause": "Readiness did not converge before timeout.",
                        "evidence": [
                            {
                                "path": "generated/sunbeam/output.log",
                                "line": 1,
                                "excerpt": "wait timed out",
                            }
                        ],
                        "candidate_mechanisms": [],
                        "recommendations": [],
                        "unknowns": [],
                    }
                )
            ),
        ]
    )

    client = OpenRouterClient(_config(), sdk_client=sdk)
    report = client.diagnose(
        "evidence text",
        session_id="uuid",
        artifact_root=artifact_root,
    )

    first_call = sdk.chat.calls[0]
    second_call = sdk.chat.calls[1]
    assert first_call["tools"][0]["function"]["name"] == "list_artifact_files"
    assert first_call["parallel_tool_calls"] is False
    assert report.summary == "The deploy step timed out."
    tool_messages = [
        message for message in second_call["messages"] if message["role"] == "tool"
    ]
    assert tool_messages[0]["tool_call_id"] == "call-1"
    assert "generated/sunbeam/output.log" in tool_messages[0]["content"]
    assert tool_messages[1]["tool_call_id"] == "call-2"
    assert "wait timed out" in tool_messages[1]["content"]
    assert sdk.chat.calls[0]["tools"] == client.exchanges[0]["request"]["tools"]
    assert client.exchanges[0]["response"]["tool_calls"][0]["id"] == "call-1"
    assert "wait timed out" in client.exchanges[1]["request"]["messages"][-1]["content"]
    system_prompt = first_call["messages"][0]["content"]
    assert "earliest event that explains the final failure surface" in system_prompt
    assert "A workload process crash is not sufficient evidence" in system_prompt
    assert "first meaningful error" not in system_prompt


def test_openrouter_client_deduplicates_repeated_tool_calls(tmp_path):
    artifact_root = tmp_path / "uuid"
    output = artifact_root / "generated/sunbeam/output.log"
    output.parent.mkdir(parents=True)
    output.write_text("wait timed out\n", encoding="utf-8")
    sdk = FakeSdkClient(
        [
            FakeSdkResponse(
                "",
                tool_calls=[
                    FakeToolCall(
                        "call-1",
                        "get_artifact_file",
                        json.dumps({"path": "generated/sunbeam/output.log"}),
                    ),
                    FakeToolCall(
                        "call-2",
                        "get_artifact_file",
                        json.dumps({"path": "generated/sunbeam/output.log"}),
                    ),
                ],
            ),
            FakeSdkResponse("I have enough context."),
        ]
    )

    OpenRouterClient(_config(), sdk_client=sdk).chat(
        "context",
        [],
        artifact_root=artifact_root,
    )

    tool_messages = [
        message for message in sdk.chat.calls[1]["messages"] if message["role"] == "tool"
    ]
    assert "wait timed out" in tool_messages[0]["content"]
    duplicate = json.loads(tool_messages[1]["content"])
    assert duplicate == {
        "duplicate_tool_call": True,
        "ok": False,
        "reason": "Duplicate tool call skipped; use earlier result for same arguments.",
    }


def test_openrouter_client_caps_tool_calls_per_round(tmp_path):
    artifact_root = tmp_path / "uuid"
    generated = artifact_root / "generated"
    generated.mkdir(parents=True)
    for index in range(5):
        (generated / f"{index}.log").write_text(f"log {index}\n", encoding="utf-8")
    sdk = FakeSdkClient(
        [
            FakeSdkResponse(
                "",
                tool_calls=[
                    FakeToolCall(
                        f"call-{index}",
                        "get_artifact_file",
                        json.dumps({"path": f"generated/{index}.log"}),
                    )
                    for index in range(5)
                ],
            ),
            FakeSdkResponse("I have enough context."),
        ]
    )

    OpenRouterClient(_config(), sdk_client=sdk).chat(
        "context",
        [],
        artifact_root=artifact_root,
    )

    tool_messages = [
        message for message in sdk.chat.calls[1]["messages"] if message["role"] == "tool"
    ]
    assert "log 0" in tool_messages[0]["content"]
    assert "log 3" in tool_messages[3]["content"]
    capped = json.loads(tool_messages[4]["content"])
    assert capped == {
        "ok": False,
        "round_tool_limit_reached": True,
        "reason": "Tool call skipped because the per-round limit is 4.",
    }


def test_openrouter_client_emits_progress_for_tool_loop(tmp_path):
    artifact_root = tmp_path / "uuid"
    output = artifact_root / "generated/sunbeam/output.log"
    output.parent.mkdir(parents=True)
    output.write_text("wait timed out\nretry failed", encoding="utf-8")
    sdk = FakeSdkClient(
        [
            FakeSdkResponse(
                "",
                usage=FakeUsage(),
                tool_calls=[
                    FakeToolCall(
                        "call-1",
                        "get_artifact_file",
                        json.dumps({"path": "generated/sunbeam/output.log"}),
                    ),
                ],
            ),
            FakeSdkResponse(
                json.dumps(
                    {
                        "summary": "The deploy step timed out.",
                        "failure_surface": "sunbeam cluster resize timed out.",
                        "confidence": "supported",
                        "root_cause": "Readiness did not converge before timeout.",
                        "evidence": [
                            {
                                "path": "generated/sunbeam/output.log",
                                "line": 1,
                                "excerpt": "wait timed out",
                            }
                        ],
                        "candidate_mechanisms": [],
                        "recommendations": [],
                        "unknowns": [],
                    }
                ),
                usage=FakeUsage(),
            ),
        ]
    )
    client = OpenRouterClient(_config(), sdk_client=sdk)
    events = []

    client.diagnose(
        "evidence text",
        session_id="uuid-diagnosis",
        artifact_root=artifact_root,
        max_tool_rounds=2,
        progress=events.append,
    )

    traces = [event.to_trace() for event in events]
    assert [event["phase"] for event in traces] == [
        "model_request",
        "tool_call",
        "tool_result",
        "model_request",
        "completed",
    ]
    assert traces[1]["tool_name"] == "get_artifact_file"
    assert traces[1]["target"] == "generated/sunbeam/output.log"
    assert traces[2]["result_chars"] > 0
    assert traces[-1]["total_tokens"] == 120
    assert traces[-1]["total_cost"] == 0.00123


def test_openrouter_client_executes_artifact_tool_calls_for_chat(tmp_path):
    artifact_root = tmp_path / "uuid"
    log = artifact_root / "generated/run.log"
    log.parent.mkdir(parents=True)
    log.write_text("failure detail", encoding="utf-8")
    sdk = FakeSdkClient(
        [
            FakeSdkResponse(
                "",
                tool_calls=[
                    FakeToolCall(
                        "call-1",
                        "get_artifact_file",
                        json.dumps({"path": "generated/run.log"}),
                    )
                ],
            ),
            FakeSdkResponse("The failure detail confirms the timeout."),
        ]
    )

    answer = OpenRouterClient(_config(), sdk_client=sdk).chat(
        "context",
        [{"role": "user", "content": "Need more detail"}],
        session_id="uuid",
        artifact_root=artifact_root,
    )

    assert answer == "The failure detail confirms the timeout."
    assert sdk.chat.calls[0]["tools"][1]["function"]["name"] == "get_artifact_file"
    assert sdk.chat.calls[0]["parallel_tool_calls"] is False
    tool_messages = [
        message for message in sdk.chat.calls[1]["messages"] if message["role"] == "tool"
    ]
    assert "failure detail" in tool_messages[-1]["content"]


def test_openrouter_client_trims_large_tool_results_by_budget(tmp_path):
    artifact_root = tmp_path / "uuid"
    log = artifact_root / "generated/run.log"
    log.parent.mkdir(parents=True)
    log.write_text("x" * 500, encoding="utf-8")
    sdk = FakeSdkClient(
        [
            FakeSdkResponse(
                "",
                tool_calls=[
                    FakeToolCall(
                        "call-1",
                        "get_artifact_file",
                        json.dumps({"path": "generated/run.log", "max_bytes": 500}),
                    )
                ],
            ),
            FakeSdkResponse("I have enough context."),
        ]
    )

    answer = OpenRouterClient(_config(), sdk_client=sdk).chat(
        "context",
        [],
        artifact_root=artifact_root,
        max_tool_result_chars=180,
    )

    tool_messages = [
        message for message in sdk.chat.calls[1]["messages"] if message["role"] == "tool"
    ]
    tool_result = json.loads(tool_messages[-1]["content"])
    assert answer == "I have enough context."
    assert tool_result["ok"] is True
    assert tool_result["tool_result_truncated_by_budget"] is True
    assert len(tool_result["content"]) < 100
    assert len(tool_messages[-1]["content"]) <= 180


def test_openrouter_client_applies_tool_result_budget_across_calls(tmp_path):
    artifact_root = tmp_path / "uuid"
    first = artifact_root / "generated/first.log"
    second = artifact_root / "generated/second.log"
    first.parent.mkdir(parents=True)
    first.write_text("a" * 300, encoding="utf-8")
    second.write_text("b" * 300, encoding="utf-8")
    sdk = FakeSdkClient(
        [
            FakeSdkResponse(
                "",
                tool_calls=[
                    FakeToolCall(
                        "call-1",
                        "get_artifact_file",
                        json.dumps({"path": "generated/first.log", "max_bytes": 300}),
                    ),
                    FakeToolCall(
                        "call-2",
                        "get_artifact_file",
                        json.dumps({"path": "generated/second.log", "max_bytes": 300}),
                    ),
                ],
            ),
            FakeSdkResponse("I have enough context."),
        ]
    )

    OpenRouterClient(_config(), sdk_client=sdk).chat(
        "context",
        [],
        artifact_root=artifact_root,
        max_tool_result_chars=260,
    )

    tool_messages = [
        message
        for message in sdk.chat.calls[1]["messages"]
        if message.get("role") == "tool"
    ]
    assert sum(len(message["content"]) for message in tool_messages) <= 260
    assert json.loads(tool_messages[-1]["content"])["tool_result_truncated_by_budget"] is True


def test_openrouter_client_reports_truncation_when_call_budget_is_tiny(tmp_path):
    artifact_root = tmp_path / "uuid"
    first = artifact_root / "generated/first.log"
    second = artifact_root / "generated/second.log"
    first.parent.mkdir(parents=True)
    first.write_text("a" * 300, encoding="utf-8")
    second.write_text("b" * 300, encoding="utf-8")
    sdk = FakeSdkClient(
        [
            FakeSdkResponse(
                "",
                tool_calls=[
                    FakeToolCall(
                        "call-1",
                        "get_artifact_file",
                        json.dumps({"path": "generated/first.log", "max_bytes": 300}),
                    ),
                    FakeToolCall(
                        "call-2",
                        "get_artifact_file",
                        json.dumps({"path": "generated/second.log", "max_bytes": 300}),
                    ),
                ],
            ),
            FakeSdkResponse("I have enough context."),
        ]
    )

    OpenRouterClient(_config(), sdk_client=sdk).chat(
        "context",
        [],
        artifact_root=artifact_root,
        max_tool_result_chars=20,
    )

    tool_messages = [
        message
        for message in sdk.chat.calls[1]["messages"]
        if message.get("role") == "tool"
    ]
    assert all(
        "tool_result_truncated_by_budget" in message["content"]
        for message in tool_messages
    )


def test_openrouter_client_retries_diagnosis_when_needs_more_evidence_without_tools(
    tmp_path,
):
    artifact_root = tmp_path / "uuid"
    log = artifact_root / "generated/sunbeam/validation_refstack.log"
    log.parent.mkdir(parents=True)
    log.write_text("Details: Unexpected status code 502\n", encoding="utf-8")
    incomplete = {
        "summary": "The refstack command failed.",
        "failure_surface": "sunbeam validation run refstack exited 1.",
        "confidence": "unknown",
        "root_cause": "",
        "needs_more_evidence": True,
        "evidence": [],
        "candidate_mechanisms": [],
        "recommendations": ["Examine the validation_refstack log."],
        "unknowns": ["What specific tests failed?"],
    }
    complete = {
        "summary": "Refstack failed while authenticating for volume tests.",
        "failure_surface": "One Tempest volume test class failed during setup.",
        "confidence": "supported",
        "root_cause": "Keystone returned HTTP 502 during Tempest auth.",
        "needs_more_evidence": False,
        "evidence": [
            {
                "path": "generated/sunbeam/validation_refstack.log",
                "line": 1,
                "excerpt": "Details: Unexpected status code 502",
            }
        ],
        "candidate_mechanisms": [],
        "recommendations": [],
        "unknowns": [],
    }
    sdk = FakeSdkClient(
        [
            FakeSdkResponse(json.dumps(incomplete)),
            FakeSdkResponse(
                "",
                tool_calls=[
                    FakeToolCall(
                        "call-1",
                        "search_artifacts",
                        json.dumps({"pattern": "Unexpected status code 502"}),
                    )
                ],
            ),
            FakeSdkResponse(json.dumps(complete)),
        ]
    )

    report = OpenRouterClient(_config(), sdk_client=sdk).diagnose(
        "evidence text",
        session_id="uuid",
        artifact_root=artifact_root,
    )

    assert report.root_cause == "Keystone returned HTTP 502 during Tempest auth."
    assert report.needs_more_evidence is False
    assert len(sdk.chat.calls) == 3
    assert sdk.chat.calls[0]["tool_choice"] == "auto"
    assert sdk.chat.calls[1]["tool_choice"] == "required"
    retry_message = sdk.chat.calls[1]["messages"][-1]["content"]
    assert "use the artifact tools" in retry_message
    assert "Do not answer yet" in retry_message


def test_openrouter_client_retries_supported_no_tool_diagnosis_until_evidence_tool_used(
    tmp_path,
):
    artifact_root = tmp_path / "uuid"
    log = artifact_root / "generated/sunbeam/validation_refstack.log"
    log.parent.mkdir(parents=True)
    log.write_text("Details: Unexpected status code 502\n", encoding="utf-8")
    unsupported = {
        "summary": "Refstack failed.",
        "failure_surface": "refstack exited 1.",
        "confidence": "supported",
        "root_cause": "Keystone returned HTTP 502 during Tempest auth.",
        "needs_more_evidence": False,
        "evidence": [
            {
                "path": "generated/sunbeam/validation_refstack.log",
                "line": 1,
                "excerpt": "Details: Unexpected status code 502",
            }
        ],
        "candidate_mechanisms": [],
        "recommendations": [],
        "unknowns": [],
    }
    supported = {
        **unsupported,
        "summary": "Refstack failed while authenticating.",
    }
    sdk = FakeSdkClient(
        [
            FakeSdkResponse(json.dumps(unsupported)),
            FakeSdkResponse(
                "",
                tool_calls=[
                    FakeToolCall(
                        "call-1",
                        "search_artifacts",
                        json.dumps({"pattern": "Unexpected status code 502"}),
                    )
                ],
            ),
            FakeSdkResponse(json.dumps(supported)),
        ]
    )

    report = OpenRouterClient(_config(), sdk_client=sdk).diagnose(
        "evidence text",
        artifact_root=artifact_root,
    )

    assert report.root_cause == "Keystone returned HTTP 502 during Tempest auth."
    assert len(sdk.chat.calls) == 3
    assert sdk.chat.calls[0]["tool_choice"] == "auto"
    assert sdk.chat.calls[1]["tool_choice"] == "required"
    retry_message = sdk.chat.calls[1]["messages"][-1]["content"]
    assert "supported or confirmed diagnosis" in retry_message
    assert "evidence-producing artifact tool" in retry_message


def test_openrouter_client_rejects_supported_diagnosis_when_required_tools_are_ignored(
    tmp_path,
):
    response = {
        "summary": "Refstack failed.",
        "failure_surface": "refstack exited 1.",
        "confidence": "supported",
        "root_cause": "Keystone returned HTTP 502 during Tempest auth.",
        "needs_more_evidence": False,
        "evidence": [
            {
                "path": "generated/sunbeam/validation_refstack.log",
                "line": 1,
                "excerpt": "Details: Unexpected status code 502",
            }
        ],
        "candidate_mechanisms": [
            {
                "name": "keystone 502",
                "status": "supported",
                "rationale": "The provided evidence says 502.",
            }
        ],
        "recommendations": [],
        "unknowns": [],
    }
    sdk = FakeSdkClient(
        [
            FakeSdkResponse(json.dumps(response)),
            FakeSdkResponse(json.dumps(response)),
        ]
    )

    with pytest.raises(RuntimeError, match="ignored required artifact tool use"):
        OpenRouterClient(_config(), sdk_client=sdk).diagnose(
            "evidence text",
            artifact_root=tmp_path,
        )

    assert len(sdk.chat.calls) == 2


def test_openrouter_client_rejects_discovery_only_supported_diagnosis_without_evidence_tool(
    tmp_path,
):
    response = {
        "summary": "Refstack failed.",
        "failure_surface": "refstack exited 1.",
        "confidence": "confirmed",
        "root_cause": "Keystone returned HTTP 502 during Tempest auth.",
        "needs_more_evidence": False,
        "evidence": [],
        "candidate_mechanisms": [
            {
                "name": "keystone 502",
                "status": "confirmed",
                "rationale": "A candidate artifact was listed.",
            }
        ],
        "recommendations": [],
        "unknowns": [],
    }
    sdk = FakeSdkClient(
        [
            FakeSdkResponse(
                "",
                tool_calls=[FakeToolCall("call-1", "list_artifact_files", "{}")],
            ),
            FakeSdkResponse(json.dumps(response)),
            FakeSdkResponse(
                "",
                tool_calls=[FakeToolCall("call-2", "list_sosreports", "{}")],
            ),
            FakeSdkResponse(json.dumps(response)),
        ]
    )

    with pytest.raises(RuntimeError, match="evidence-producing artifact tool"):
        OpenRouterClient(_config(), sdk_client=sdk).diagnose(
            "evidence text",
            artifact_root=tmp_path,
        )

    assert len(sdk.chat.calls) == 4


def test_openrouter_client_does_not_retry_after_diagnosis_used_tools(tmp_path):
    artifact_root = tmp_path / "uuid"
    log = artifact_root / "generated/sunbeam/validation_refstack.log"
    log.parent.mkdir(parents=True)
    log.write_text("Details: Unexpected status code 502\n", encoding="utf-8")
    response = {
        "summary": "The refstack command failed.",
        "failure_surface": "sunbeam validation run refstack exited 1.",
        "confidence": "unknown",
        "root_cause": "",
        "needs_more_evidence": True,
        "evidence": [],
        "candidate_mechanisms": [],
        "recommendations": ["Review service logs manually."],
        "unknowns": ["No decisive service-side cause found."],
    }
    sdk = FakeSdkClient(
        [
            FakeSdkResponse(
                "",
                tool_calls=[
                    FakeToolCall(
                        "call-1",
                        "search_artifacts",
                        json.dumps({"pattern": "Unexpected status code 502"}),
                    )
                ],
            ),
            FakeSdkResponse(json.dumps(response)),
        ]
    )

    report = OpenRouterClient(_config(), sdk_client=sdk).diagnose(
        "evidence text",
        artifact_root=artifact_root,
    )

    assert report.needs_more_evidence is True
    assert len(sdk.chat.calls) == 2


def test_openrouter_client_downgrades_confirmed_diagnosis_after_tool_budget_fallback(
    tmp_path,
):
    sdk = FakeSdkClient(
        [
            FakeSdkResponse(
                "",
                tool_calls=[FakeToolCall("call-1", "list_artifact_files", "{}")],
            ),
            FakeSdkResponse(
                json.dumps(
                    {
                        "summary": "The join timed out waiting for k8s.",
                        "failure_surface": "sunbeam cluster join timed out.",
                        "confidence": "confirmed",
                        "root_cause": "The orchestrator agent was lost.",
                        "needs_more_evidence": False,
                        "evidence": [],
                        "candidate_mechanisms": [
                            {
                                "name": "orchestrator agent lost",
                                "status": "confirmed",
                                "rationale": "Status showed the unit lost.",
                            }
                        ],
                        "recommendations": ["Inspect the sosreport."],
                        "unknowns": [],
                    }
                )
            ),
        ]
    )

    report = OpenRouterClient(_config(), sdk_client=sdk).diagnose(
        "evidence text",
        artifact_root=tmp_path,
        max_tool_rounds=1,
    )

    assert report.confidence == "supported"
    assert report.needs_more_evidence is True
    assert report.candidate_mechanisms[0].status == "supported"
    assert any("tool budget" in unknown for unknown in report.unknowns)


def test_openrouter_client_nudges_targeted_read_after_broad_tool_only_round(
    tmp_path,
):
    log = tmp_path / "generated/sunbeam/output.log"
    log.parent.mkdir(parents=True)
    log.write_text("wait timed out\n", encoding="utf-8")
    sdk = FakeSdkClient(
        [
            FakeSdkResponse(
                "",
                tool_calls=[
                    FakeToolCall(
                        "call-1",
                        "search_artifacts",
                        json.dumps({"pattern": "wait timed out"}),
                    )
                ],
            ),
            FakeSdkResponse("The timeout is visible."),
        ]
    )

    OpenRouterClient(_config(), sdk_client=sdk).chat(
        "context",
        [],
        artifact_root=tmp_path,
    )

    followup_messages = sdk.chat.calls[1]["messages"]
    user_messages = [
        message for message in followup_messages if message["role"] == "user"
    ]
    assert any("targeted read" in message["content"] for message in user_messages)
    assert any("get_artifact_file" in message["content"] for message in user_messages)


def test_openrouter_client_executes_sosreport_tool_calls_for_chat(tmp_path):
    artifact_root = tmp_path / "uuid"
    archive = artifact_root / "generated/sunbeam/sosreport-node-a-2026-06-23-abc.tar.xz"
    _write_tar_member(
        archive,
        "sosreport-node-a-2026-06-23-abc/home/ubuntu/snap/openstack/common/logs/sunbeam.log",
        "starting\nResultType.COMPLETED\n",
    )
    sdk = FakeSdkClient(
        [
            FakeSdkResponse(
                "",
                tool_calls=[
                    FakeToolCall(
                        "call-1",
                        "search_sosreport",
                        json.dumps(
                            {
                                "archive_path": (
                                    "generated/sunbeam/"
                                    "sosreport-node-a-2026-06-23-abc.tar.xz"
                                ),
                                "pattern": "ResultType.COMPLETED",
                                "prefix": "home/ubuntu/snap/openstack/common/logs/",
                            }
                        ),
                    )
                ],
            ),
            FakeSdkResponse("The remote command completed."),
        ]
    )

    answer = OpenRouterClient(_config(), sdk_client=sdk).chat(
        "context",
        [{"role": "user", "content": "Did the remote command complete?"}],
        session_id="uuid",
        artifact_root=artifact_root,
    )

    assert answer == "The remote command completed."
    tool_names = [tool["function"]["name"] for tool in sdk.chat.calls[0]["tools"]]
    assert "search_sosreport" in tool_names
    tool_messages = [
        message for message in sdk.chat.calls[1]["messages"] if message["role"] == "tool"
    ]
    assert "ResultType.COMPLETED" in tool_messages[-1]["content"]


def test_openrouter_client_answers_without_tools_after_max_tool_rounds(tmp_path):
    sdk = FakeSdkClient(
        [
            FakeSdkResponse(
                "",
                tool_calls=[FakeToolCall("call-1", "list_artifact_files", "{}")],
            ),
            FakeSdkResponse("Answering with the available context."),
        ]
    )

    answer = OpenRouterClient(_config(), sdk_client=sdk).chat(
        "context",
        [],
        artifact_root=tmp_path,
        max_tool_rounds=1,
    )

    assert answer == "Answering with the available context."
    assert sdk.chat.calls[0]["tools"][0]["function"]["name"] == "list_artifact_files"
    assert "tools" in sdk.chat.calls[1]
    assert sdk.chat.calls[1]["tool_choice"] == "none"
    assert sdk.chat.calls[1]["parallel_tool_calls"] is False
    assert "budget_exhausted" in sdk.chat.calls[1]["messages"][-1]["content"]


def test_openrouter_client_defaults_to_twelve_tool_rounds_before_partial_final(
    tmp_path,
):
    responses = [
        FakeSdkResponse(
            "",
            tool_calls=[FakeToolCall(f"call-{index}", "list_artifact_files", "{}")],
        )
        for index in range(12)
    ]
    responses.append(FakeSdkResponse("Answering with partial findings."))
    sdk = FakeSdkClient(responses)

    answer = OpenRouterClient(_config(), sdk_client=sdk).chat(
        "context",
        [],
        artifact_root=tmp_path,
        triage_options=TriageLoopOptions(
            max_rounds=12,
            hard_max_rounds=20,
            stall_limit=20,
        ),
    )

    assert answer == "Answering with partial findings."
    assert len(sdk.chat.calls) == 13
    assert sdk.chat.calls[-1]["tool_choice"] == "none"
    assert "tools" in sdk.chat.calls[-1]
    assert "budget_exhausted" in sdk.chat.calls[-1]["messages"][-1]["content"]


def test_openrouter_client_finalizes_after_stalled_tool_progress(tmp_path):
    responses = [
        FakeSdkResponse(
            "",
            tool_calls=[FakeToolCall(f"call-{index}", "list_artifact_files", "{}")],
        )
        for index in range(3)
    ]
    responses.append(FakeSdkResponse("Answering after stall."))
    sdk = FakeSdkClient(responses)

    answer = OpenRouterClient(_config(), sdk_client=sdk).chat(
        "context",
        [],
        artifact_root=tmp_path,
        max_tool_rounds=12,
    )

    assert answer == "Answering after stall."
    assert len(sdk.chat.calls) == 4
    assert sdk.chat.calls[-1]["tool_choice"] == "none"
    assert "stalled" in sdk.chat.calls[-1]["messages"][-1]["content"]


def _write_tar_member(path, name, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    with tarfile.open(path, "w:xz") as archive:
        archive.addfile(info, io.BytesIO(data))
