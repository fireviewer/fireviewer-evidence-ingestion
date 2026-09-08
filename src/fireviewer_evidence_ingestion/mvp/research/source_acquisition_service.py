"""HTTP entrypoint for the scale-to-zero Azure CPU source worker."""

from __future__ import annotations

import json
import os
from hmac import compare_digest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import monotonic
from typing import Any, Protocol

from pydantic import Field, SecretStr, model_validator

from fireviewer_contracts.contracts import SafeIdentifierV2, StrictModel
from fireviewer_evidence_ingestion.mvp.research.multimodal_evidence import (
    AzureFederatedBedrockClient,
    AzureManagedIdentityWebTokenProvider,
    BedrockPixtralConfig,
    BedrockPixtralMultimodalProvider,
    InvocationLimitedBedrockClient,
)
from fireviewer_evidence_ingestion.mvp.research.source_acquisition import (
    CpuSourceAcquisitionWorker,
    SourceAcquisitionPlan,
    SourceAcquisitionRunReceipt,
)
from fireviewer_evidence_ingestion.mvp.research.source_planner import (
    AutomaticSourceAcquisitionPlanner,
    AutomaticSourcePlannerConfig,
)
from fireviewer_contracts.mvp.supervision.backend_event_evidence import (
    AzureBackendEventEvidenceAdapter,
    AzureBackendEventEvidenceConfig,
    AzureBackendIncidentDayEvidenceAdapter,
    BackendIncidentDayResearchPublisher,
    BackendResearchEvidencePublisher,
    EventEvidenceRepository,
)
from fireviewer_evidence_ingestion.research_broker import ResearchBroker

_MAX_REQUEST_BYTES = 4 * 1_024


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


class SourceAcquisitionRequest(StrictModel):
    candidate_id: SafeIdentifierV2 | None = None
    analysis_id: SafeIdentifierV2 | None = None

    @model_validator(mode="after")
    def validate_target(self) -> SourceAcquisitionRequest:
        if (self.candidate_id is None) == (self.analysis_id is None):
            raise ValueError("exactly one durable acquisition target is required")
        return self


class SourceAcquisitionRunner(Protocol):
    def run_candidate(self, candidate_id: str) -> dict[str, Any]: ...

    def run_analysis(self, analysis_id: str) -> dict[str, Any]: ...


class SourceAcquisitionOrchestrator:
    def __init__(
        self,
        *,
        repository: EventEvidenceRepository,
        incident_repository: EventEvidenceRepository,
        planner: AutomaticSourceAcquisitionPlanner,
        worker: CpuSourceAcquisitionWorker,
        incident_worker: CpuSourceAcquisitionWorker,
        max_incident_cycles: int = 1,
        max_incident_runtime_seconds: float = 180,
    ) -> None:
        self._repository = repository
        self._incident_repository = incident_repository
        self._planner = planner
        self._worker = worker
        self._incident_worker = incident_worker
        self._max_incident_cycles = max_incident_cycles
        self._max_incident_runtime_seconds = max_incident_runtime_seconds

    @staticmethod
    def _result(
        *,
        receipt: SourceAcquisitionRunReceipt,
        plan: SourceAcquisitionPlan,
        planner: str,
    ) -> dict[str, Any]:
        return {
            **receipt.model_dump(mode="json"),
            "planner": planner,
            "query_count": len(plan.queries),
            "domain_policy_count": len(plan.source_policies),
        }

    def run_candidate(self, candidate_id: str) -> dict[str, Any]:
        durable = self._repository.read(candidate_id)
        plan = self._planner.build(durable)
        receipt = self._worker.run(plan)
        return self._result(
            receipt=receipt,
            plan=plan,
            planner="automatic-durable-candidate-v1",
        )

    def run_analysis(self, analysis_id: str) -> dict[str, Any]:
        started_at = monotonic()
        cycles = 0
        executed_waves: list[int] = []
        receipt: SourceAcquisitionRunReceipt | None = None
        plan: SourceAcquisitionPlan | None = None
        durable = self._incident_repository.read(analysis_id)
        while cycles < self._max_incident_cycles:
            if monotonic() - started_at >= self._max_incident_runtime_seconds:
                break
            plan = self._planner.build(durable)
            receipt = self._incident_worker.run(plan)
            cycles += 1
            if not executed_waves or executed_waves[-1] != receipt.wave_number:
                executed_waves.append(receipt.wave_number)
            durable = self._incident_repository.read(analysis_id)
            coverage = durable.incident_day_coverage
            if (
                coverage is not None
                and coverage.documentary_ready
                and receipt.completed
                and receipt.converged
            ):
                break
            if receipt.safety_limit_reached or receipt.wave_number >= 16:
                break
        if receipt is None or plan is None:
            raise RuntimeError("incident-day acquisition has no executable research work")
        result = self._result(
            receipt=receipt,
            plan=plan,
            planner="automatic-incident-day-v1",
        )
        coverage = durable.incident_day_coverage
        result.update(
            {
                "orchestration_cycles": cycles,
                "waves_executed": executed_waves,
                "orchestration_complete": bool(
                    coverage is not None
                    and coverage.documentary_ready
                    and receipt.completed
                    and receipt.converged
                ),
                "daily_bundle_ready": bool(coverage is not None and coverage.coverage_ready),
                "coverage_ready": bool(coverage is not None and coverage.documentary_ready),
                "missing_dimensions": (
                    list(coverage.missing_dimensions) if coverage is not None else []
                ),
            }
        )
        result["analysis_id"] = result.pop("candidate_id")
        return result


class SourceAcquisitionServiceSettings(StrictModel):
    host: str = Field(default="0.0.0.0", min_length=1, max_length=255)  # noqa: S104
    port: int = Field(default=8080, ge=1, le=65_535)
    worker_token: SecretStr = Field(min_length=32, max_length=4_096)
    broker_control_token: SecretStr = Field(min_length=32, max_length=4_096)
    managed_identity_client_id: str = Field(min_length=36, max_length=36)
    aws_role_arn: str = Field(pattern=r"^arn:aws:iam::\d{12}:role/[A-Za-z0-9+=,.@_/-]+$")
    aws_oidc_audience: str = Field(min_length=8, max_length=512)
    aws_region: str = Field(default="eu-west-3", pattern=r"^[a-z]{2}-[a-z]+-\d$")
    bedrock_model_id: str = Field(
        default="eu.mistral.pixtral-large-2502-v1:0",
        min_length=3,
        max_length=256,
    )
    multimodal_enabled: bool = False
    bedrock_paid_invocation_enabled: bool = False
    authorized_bedrock_invocations_per_run: int = Field(default=0, ge=0, le=4)
    bedrock_maximum_output_tokens: int = Field(default=2_048, ge=256, le=4_096)
    max_incident_cycles: int = Field(default=1, ge=1, le=8)
    max_incident_runtime_seconds: int = Field(default=180, ge=60, le=210)
    max_pages_per_run: int = Field(default=1, ge=1, le=5)
    results_per_page: int = Field(default=1, ge=1, le=5)
    media_per_source: int = Field(default=2, ge=1, le=8)
    max_multimodal_analyses_per_run: int = Field(default=1, ge=1, le=4)
    search_provider_domain: str = Field(default="html.duckduckgo.com", min_length=3)
    search_template: str = Field(
        default="https://html.duckduckgo.com/html/?q={query}", min_length=16
    )
    backend: AzureBackendEventEvidenceConfig

    @model_validator(mode="after")
    def validate_paid_multimodal_gate(self) -> SourceAcquisitionServiceSettings:
        if self.multimodal_enabled and not self.bedrock_paid_invocation_enabled:
            raise ValueError("multimodal Bedrock calls require the explicit paid invocation gate")
        if self.multimodal_enabled and self.authorized_bedrock_invocations_per_run < 1:
            raise ValueError("multimodal Bedrock calls require a positive invocation budget")
        if (
            self.multimodal_enabled
            and self.max_multimodal_analyses_per_run > self.authorized_bedrock_invocations_per_run
        ):
            raise ValueError("multimodal run limit exceeds the authorized Bedrock budget")
        return self

    @classmethod
    def from_env(cls) -> SourceAcquisitionServiceSettings:
        return cls(
            host=os.getenv("FIREVIEWER_SOURCE_WORKER_HOST", "0.0.0.0"),  # noqa: S104
            port=int(os.getenv("PORT", "8080")),
            worker_token=SecretStr(os.environ["FIREVIEWER_SOURCE_WORKER_TOKEN"]),
            broker_control_token=SecretStr(os.environ["FIREVIEWER_SOURCE_BROKER_CONTROL_TOKEN"]),
            managed_identity_client_id=os.environ["AZURE_CLIENT_ID"],
            aws_role_arn=os.environ["FIREVIEWER_BEDROCK_ROLE_ARN"],
            aws_oidc_audience=os.environ["FIREVIEWER_AWS_OIDC_AUDIENCE"],
            aws_region=os.getenv("AWS_REGION", "eu-west-3"),
            bedrock_model_id=os.getenv(
                "FIREVIEWER_BEDROCK_MODEL_ID",
                "eu.mistral.pixtral-large-2502-v1:0",
            ),
            multimodal_enabled=_env_bool("FIREVIEWER_MULTIMODAL_ENABLED", False),
            bedrock_paid_invocation_enabled=_env_bool(
                "FIREVIEWER_BEDROCK_PAID_INVOCATION_ENABLED", False
            ),
            authorized_bedrock_invocations_per_run=int(
                os.getenv("FIREVIEWER_BEDROCK_AUTHORIZED_INVOCATIONS_PER_RUN", "0")
            ),
            bedrock_maximum_output_tokens=int(
                os.getenv("FIREVIEWER_BEDROCK_MAX_OUTPUT_TOKENS", "2048")
            ),
            max_incident_cycles=int(os.getenv("FIREVIEWER_SOURCE_MAX_INCIDENT_CYCLES", "1")),
            max_incident_runtime_seconds=int(
                os.getenv("FIREVIEWER_SOURCE_MAX_RUNTIME_SECONDS", "180")
            ),
            max_pages_per_run=int(os.getenv("FIREVIEWER_SOURCE_MAX_PAGES_PER_RUN", "1")),
            results_per_page=int(os.getenv("FIREVIEWER_SOURCE_RESULTS_PER_PAGE", "1")),
            media_per_source=int(os.getenv("FIREVIEWER_SOURCE_MEDIA_PER_SOURCE", "2")),
            max_multimodal_analyses_per_run=int(
                os.getenv("FIREVIEWER_SOURCE_MAX_MULTIMODAL_PER_RUN", "1")
            ),
            search_provider_domain=os.getenv(
                "FIREVIEWER_SOURCE_SEARCH_PROVIDER_DOMAIN", "html.duckduckgo.com"
            ),
            search_template=os.getenv(
                "FIREVIEWER_SOURCE_SEARCH_TEMPLATE",
                "https://html.duckduckgo.com/html/?q={query}",
            ),
            backend=AzureBackendEventEvidenceConfig(
                base_url=os.environ["FIREVIEWER_BACKEND_BASE_URL"],
                bearer_token=SecretStr(os.environ["FIREVIEWER_BACKEND_TOKEN"]),
                timeout_seconds=float(os.getenv("FIREVIEWER_BACKEND_TIMEOUT_SECONDS", "20")),
            ),
        )


class SourceAcquisitionService:
    def __init__(
        self,
        *,
        settings: SourceAcquisitionServiceSettings,
        runner: SourceAcquisitionRunner | None = None,
    ) -> None:
        self.settings = settings
        if runner is None:
            repository = AzureBackendEventEvidenceAdapter(settings.backend)
            incident_repository = AzureBackendIncidentDayEvidenceAdapter(settings.backend)
            provider = None
            if settings.multimodal_enabled:
                config = BedrockPixtralConfig(
                    region_name=settings.aws_region,
                    model_id=settings.bedrock_model_id,
                    maximum_output_tokens=settings.bedrock_maximum_output_tokens,
                )
                provider = BedrockPixtralMultimodalProvider(
                    config,
                    client=InvocationLimitedBedrockClient(
                        AzureFederatedBedrockClient(
                            role_arn=settings.aws_role_arn,
                            region_name=settings.aws_region,
                            web_token_provider=AzureManagedIdentityWebTokenProvider(
                                audience=settings.aws_oidc_audience,
                                managed_identity_client_id=settings.managed_identity_client_id,
                            ),
                        ),
                        maximum_invocations=(settings.authorized_bedrock_invocations_per_run),
                    ),
                )
            control_token = settings.broker_control_token.get_secret_value()
            worker = CpuSourceAcquisitionWorker(
                repository=repository,
                publisher=BackendResearchEvidencePublisher(settings.backend),
                broker=ResearchBroker(control_token=control_token),
                broker_control_token=control_token,
                multimodal_evidence_provider=provider,
            )
            incident_worker = CpuSourceAcquisitionWorker(
                repository=incident_repository,
                publisher=BackendIncidentDayResearchPublisher(settings.backend),
                broker=ResearchBroker(control_token=control_token),
                broker_control_token=control_token,
                multimodal_evidence_provider=provider,
            )
            runner = SourceAcquisitionOrchestrator(
                repository=repository,
                incident_repository=incident_repository,
                planner=AutomaticSourceAcquisitionPlanner(
                    AutomaticSourcePlannerConfig(
                        search_provider_domain=settings.search_provider_domain,
                        search_template=settings.search_template,
                        results_per_page=settings.results_per_page,
                        media_per_source=settings.media_per_source,
                        max_pages_per_run=settings.max_pages_per_run,
                        max_multimodal_analyses_per_run=(settings.max_multimodal_analyses_per_run),
                    )
                ),
                worker=worker,
                incident_worker=incident_worker,
                max_incident_cycles=settings.max_incident_cycles,
                max_incident_runtime_seconds=settings.max_incident_runtime_seconds,
            )
        self.runner = runner

    def authorize(self, header: str | None) -> bool:
        expected = f"Bearer {self.settings.worker_token.get_secret_value()}"
        return header is not None and compare_digest(header, expected)

    def run(self, payload: object) -> dict[str, Any]:
        request = SourceAcquisitionRequest.model_validate(payload)
        if request.analysis_id is not None:
            return self.runner.run_analysis(request.analysis_id)
        if request.candidate_id is None:
            raise RuntimeError("validated source target is missing")
        return self.runner.run_candidate(request.candidate_id)


def _handler_for(service: SourceAcquisitionService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "FireViewerSourceCpu/2.0"

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/healthz":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self._json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "runtime": "azure-cpu",
                    "planner": "automatic-durable-target-v1",
                    "incident_day_supported": True,
                    "manual_plan_accepted": False,
                    "multimodal_provider": "aws-bedrock-pixtral",
                    "multimodal_model": service.settings.bedrock_model_id,
                    "multimodal_enabled": service.settings.multimodal_enabled,
                    "paid_invocation_enabled": (service.settings.bedrock_paid_invocation_enabled),
                    "authorized_invocations_per_run": (
                        service.settings.authorized_bedrock_invocations_per_run
                    ),
                    "multimodal_input_scope": "public-web-only",
                    "raw_scraped_content_stored": False,
                    "transcripts_stored": False,
                    "public_media_binaries_stored": False,
                },
            )

        def do_POST(self) -> None:
            if self.path not in {
                "/v1/event-evidence/research",
                "/v1/incident-day/research",
            }:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            if not service.authorize(self.headers.get("Authorization")):
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= _MAX_REQUEST_BYTES:
                    raise ValueError("request_size_invalid")
                payload = json.loads(self.rfile.read(length))
                if self.path == "/v1/incident-day/research" and (
                    not isinstance(payload, dict) or "analysis_id" not in payload
                ):
                    raise ValueError("analysis_id is required for this route")
                if self.path == "/v1/event-evidence/research" and (
                    not isinstance(payload, dict) or "candidate_id" not in payload
                ):
                    raise ValueError("candidate_id is required for this route")
                result = service.run(payload)
            except Exception as exc:
                self._json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": type(exc).__name__, "detail": str(exc)[:1_000]},
                )
                return
            self._json(HTTPStatus.OK, result)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def create_source_acquisition_server(
    service: SourceAcquisitionService,
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(
        (service.settings.host, service.settings.port),
        _handler_for(service),
    )


def main() -> None:
    settings = SourceAcquisitionServiceSettings.from_env()
    server = create_source_acquisition_server(SourceAcquisitionService(settings=settings))
    server.serve_forever()


if __name__ == "__main__":
    main()


__all__ = [
    "SourceAcquisitionOrchestrator",
    "SourceAcquisitionRequest",
    "SourceAcquisitionService",
    "SourceAcquisitionServiceSettings",
    "create_source_acquisition_server",
]
