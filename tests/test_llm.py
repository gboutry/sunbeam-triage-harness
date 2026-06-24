import json

from sunbeam_triage.config import Config
from sunbeam_triage.llm import OpenRouterClient


class FakeSdkResponse:
    def __init__(self, content, usage=None):
        self.choices = [FakeChoice(content)]
        self.usage = usage


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeUsage:
    prompt_tokens = 100
    completion_tokens = 20
    total_tokens = 120
    prompt_tokens_details = {"cached_tokens": 75}
    cache_write_tokens = 25


class FakeChat:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def send(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


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
