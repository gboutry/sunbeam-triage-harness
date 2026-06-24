import json

from sunbeam_triage.config import Config
from sunbeam_triage.llm import OpenRouterClient


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
    assert second_call["messages"][-2]["role"] == "tool"
    assert second_call["messages"][-2]["tool_call_id"] == "call-1"
    assert "generated/sunbeam/output.log" in second_call["messages"][-2]["content"]
    assert second_call["messages"][-1]["role"] == "tool"
    assert second_call["messages"][-1]["tool_call_id"] == "call-2"
    assert "wait timed out" in second_call["messages"][-1]["content"]
    assert sdk.chat.calls[0]["tools"] == client.exchanges[0]["request"]["tools"]
    assert client.exchanges[0]["response"]["tool_calls"][0]["id"] == "call-1"
    assert "wait timed out" in client.exchanges[1]["request"]["messages"][-1]["content"]


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
    assert sdk.chat.calls[1]["messages"][-1]["role"] == "tool"
    assert "failure detail" in sdk.chat.calls[1]["messages"][-1]["content"]


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
    assert "tools" not in sdk.chat.calls[1]
    assert "tool_choice" not in sdk.chat.calls[1]
    assert "parallel_tool_calls" not in sdk.chat.calls[1]
    assert "tool budget is exhausted" in sdk.chat.calls[1]["messages"][-1]["content"]
