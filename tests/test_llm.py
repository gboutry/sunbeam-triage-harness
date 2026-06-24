import json

from sunbeam_triage.config import Config
from sunbeam_triage.llm import OpenRouterClient


class FakeHttp:
    def __init__(self):
        self.payload = None

    def post_json(self, url, payload, headers):
        self.payload = payload
        assert headers["Authorization"] == "Bearer token"
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
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
                        )
                    }
                }
            ]
        }


def test_openrouter_client_requests_structured_diagnosis():
    config = Config.load(None)
    config.llm.api_key = "token"
    config.llm.model = "openrouter/auto"
    http = FakeHttp()

    report = OpenRouterClient(config.llm, http=http).diagnose("evidence text")

    assert http.payload["model"] == "openrouter/auto"
    assert http.payload["response_format"]["type"] == "json_schema"
    assert report.summary == "The deploy step timed out."
    assert report.evidence[0].line == 2
