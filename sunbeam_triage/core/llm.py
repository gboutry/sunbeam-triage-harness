from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import LlmConfig
from .llm_exchanges import (
    ExchangeRecorder,
)
from .llm_exchanges import (
    response_content as _response_content,
)
from .llm_exchanges import (
    response_json as _response_json,
)
from .llm_policy import (
    diagnosis_confidence_requires_artifact_evidence,
    diagnosis_needs_required_tool_retry,
    downgrade_tool_budget_diagnosis,
    exchange_range_has_evidence_tool_calls,
    exchange_range_has_tool_budget_fallback,
    exchange_range_has_tool_calls,
)
from .llm_prompts import (
    chat_system_prompt,
    diagnosis_system_prompt,
    request_with_evidence_retry,
)
from .llm_schema import (
    REPORT_SCHEMA,
    AlternativeConsidered,
    CandidateMechanism,
    DiagnosisReport,
    ReportEvidence,
    TimelineEvent,
)
from .llm_tool_loop import ArtifactToolLoop
from .llm_transport import OpenRouterTransport, cache_kwargs
from .progress import ProgressSink
from .triage_state import TriageLoopOptions

__all__ = [
    "REPORT_SCHEMA",
    "AlternativeConsidered",
    "CandidateMechanism",
    "DiagnosisReport",
    "OpenRouterClient",
    "ReportEvidence",
    "TimelineEvent",
]


class OpenRouterClient:
    def __init__(self, config: LlmConfig, sdk_client: Any | None = None):
        self.config = config
        self.transport = OpenRouterTransport(config, sdk_client=sdk_client)
        self._exchange_recorder = ExchangeRecorder()
        self.exchanges = self._exchange_recorder.exchanges

    def diagnose(
        self,
        evidence_text: str,
        *,
        session_id: str | None = None,
        artifact_root: Path | None = None,
        max_tool_rounds: int | None = None,
        max_tool_result_chars: int = 60_000,
        triage_options: TriageLoopOptions | None = None,
        progress: ProgressSink | None = None,
        run_type: str = "diagnosis",
        contender_id: str | None = None,
    ) -> DiagnosisReport:
        request = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": diagnosis_system_prompt(),
                },
                {"role": "user", "content": evidence_text},
            ],
            "response_format": {"type": "json_schema", "json_schema": REPORT_SCHEMA},
        }
        if session_id:
            request["session_id"] = session_id
        request.update(cache_kwargs(self.config.model))
        exchange_start = len(self.exchanges)
        response = self._send_with_artifact_tools(
            request,
            artifact_root=artifact_root,
            max_tool_rounds=max_tool_rounds,
            max_tool_result_chars=max_tool_result_chars,
            triage_options=triage_options,
            progress=progress,
            run_id=session_id or self.config.model,
            run_type=run_type,
            contender_id=contender_id,
        )
        data = _response_json(response)
        has_tool_budget_fallback = exchange_range_has_tool_budget_fallback(
            self.exchanges,
            exchange_start,
        )
        if has_tool_budget_fallback:
            data = downgrade_tool_budget_diagnosis(data)
        if (
            not has_tool_budget_fallback
            and artifact_root is not None
            and diagnosis_needs_required_tool_retry(
                data,
                self.exchanges,
                exchange_start,
            )
        ):
            retry_request = request_with_evidence_retry(
                request,
                response,
                data=data,
            )
            retry_start = len(self.exchanges)
            response = self._send_with_artifact_tools(
                retry_request,
                artifact_root=artifact_root,
                max_tool_rounds=max_tool_rounds,
                max_tool_result_chars=max_tool_result_chars,
                triage_options=triage_options,
                tool_choice="required",
                progress=progress,
                run_id=session_id or self.config.model,
                run_type=run_type,
                contender_id=contender_id,
            )
            data = _response_json(response)
            if exchange_range_has_tool_budget_fallback(
                self.exchanges,
                exchange_start,
            ):
                has_tool_budget_fallback = True
                data = downgrade_tool_budget_diagnosis(data)
            elif not exchange_range_has_tool_calls(self.exchanges, retry_start):
                raise RuntimeError(
                    "Model ignored required artifact tool use; diagnosis was "
                    "not validated."
                )
        if (
            not has_tool_budget_fallback
            and artifact_root is not None
            and diagnosis_confidence_requires_artifact_evidence(data)
            and not exchange_range_has_evidence_tool_calls(
                self.exchanges,
                exchange_start,
            )
        ):
            raise RuntimeError(
                "Model returned a supported or confirmed diagnosis without "
                "using an evidence-producing artifact tool; diagnosis was not "
                "validated."
            )
        return DiagnosisReport.from_dict(data)

    def chat(
        self,
        context_text: str,
        messages: list[dict[str, str]],
        *,
        session_id: str | None = None,
        artifact_root: Path | None = None,
        max_tool_rounds: int | None = None,
        max_tool_result_chars: int = 60_000,
        triage_options: TriageLoopOptions | None = None,
        progress: ProgressSink | None = None,
        run_type: str = "followup",
        contender_id: str | None = None,
    ) -> str:
        request = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": chat_system_prompt(),
                },
                {"role": "user", "content": context_text},
                *messages,
            ],
        }
        if session_id:
            request["session_id"] = session_id
        request.update(cache_kwargs(self.config.model))
        response = self._send_with_artifact_tools(
            request,
            artifact_root=artifact_root,
            max_tool_rounds=max_tool_rounds,
            max_tool_result_chars=max_tool_result_chars,
            triage_options=triage_options,
            progress=progress,
            run_id=session_id or self.config.model,
            run_type=run_type,
            contender_id=contender_id,
        )
        return _response_content(response)

    def _send_with_artifact_tools(
        self,
        request: dict[str, Any],
        *,
        artifact_root: Path | None,
        max_tool_rounds: int | None,
        max_tool_result_chars: int,
        triage_options: TriageLoopOptions | None = None,
        tool_choice: str = "auto",
        progress: ProgressSink | None = None,
        run_id: str = "",
        run_type: str = "diagnosis",
        contender_id: str | None = None,
    ) -> Any:
        loop = ArtifactToolLoop(
            send=self._send,
            record_exchange=self._record_exchange,
        )
        return loop.run(
            request,
            artifact_root=artifact_root,
            max_tool_rounds=max_tool_rounds,
            max_tool_result_chars=max_tool_result_chars,
            triage_options=triage_options,
            tool_choice=tool_choice,
            progress=progress,
            run_id=run_id,
            run_type=run_type,
            contender_id=contender_id,
        )

    def _send(self, request: dict[str, Any]) -> Any:
        return self.transport.send(request)

    def _record_exchange(self, request: dict[str, Any], response: Any) -> None:
        self._exchange_recorder.record(request, response)
