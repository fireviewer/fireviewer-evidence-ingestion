from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from fireviewer_evidence_ingestion.mvp.research.public_media_cpu import (
    PublicMediaAnalysisRunReceipt,
)
from fireviewer_evidence_ingestion.mvp.research.public_media_cpu_service import (
    PublicMediaCpuService,
    PublicMediaCpuServiceSettings,
)
from fireviewer_contracts.mvp.supervision.backend_event_evidence import (
    AzureBackendEventEvidenceConfig,
)


def _settings(**updates) -> PublicMediaCpuServiceSettings:
    values = {
        "worker_token": SecretStr("w" * 40),
        "broker_control_token": SecretStr("b" * 40),
        "yolo_endpoint": "https://yolo.internal.example",
        "yolo_token": SecretStr("y" * 40),
        "backend": AzureBackendEventEvidenceConfig(
            base_url="https://backend.internal.example",
            bearer_token=SecretStr("t" * 40),
        ),
    }
    values.update(updates)
    return PublicMediaCpuServiceSettings(**values)


class _Worker:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run_analysis(self, analysis_id: str) -> PublicMediaAnalysisRunReceipt:
        self.calls.append(analysis_id)
        return PublicMediaAnalysisRunReceipt(
            analysis_id=analysis_id,
            source_revision_sha256="a" * 64,
            eligible_media_count=0,
            already_processed_count=0,
            attempted_count=0,
            succeeded_count=0,
            partial_count=0,
            failed_count=0,
            remaining_count=0,
        )


def test_service_runs_deterministic_media_stages_with_optional_providers_disabled() -> None:
    worker = _Worker()
    service = PublicMediaCpuService(settings=_settings(), worker=worker)  # type: ignore[arg-type]

    receipt = service.run({"analysis_id": "AN-DIE-2026-07-06"})

    assert receipt.analysis_id == "AN-DIE-2026-07-06"
    assert worker.calls == ["AN-DIE-2026-07-06"]


def test_service_runs_with_vl_while_transcription_remains_optional() -> None:
    with pytest.raises(ValidationError, match="Azure Speech configuration"):
        _settings(transcription_enabled=True)

    worker = _Worker()
    service = PublicMediaCpuService(
        settings=_settings(
            multimodal_enabled=True,
            managed_identity_client_id="1" * 36,
            aws_role_arn="arn:aws:iam::123456789012:role/fireviewer-media",
            aws_oidc_audience="api://fireviewer-aws-federation",
        ),
        worker=worker,  # type: ignore[arg-type]
    )

    receipt = service.run({"analysis_id": "AN-DIE-2026-07-06"})

    assert receipt.analysis_id == "AN-DIE-2026-07-06"
    assert worker.calls == ["AN-DIE-2026-07-06"]
