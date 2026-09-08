from __future__ import annotations

import json
import socket
import threading
from dataclasses import replace
from http.client import HTTPConnection

from fireviewer_contracts.mvp.contracts import EventEvidenceV1
from fireviewer_evidence_ingestion.mvp.research.source_acquisition import (
    SourceAcquisitionRunReceipt,
)
from fireviewer_evidence_ingestion.mvp.research.source_acquisition_service import (
    SourceAcquisitionOrchestrator,
    SourceAcquisitionService,
    SourceAcquisitionServiceSettings,
    create_source_acquisition_server,
)
from fireviewer_evidence_ingestion.mvp.research.source_planner import AutomaticSourceAcquisitionPlanner
from fireviewer_contracts.mvp.supervision.backend_event_evidence import (
    AzureBackendEventEvidenceConfig,
    BackendIncidentDayCoverage,
    DurableEventEvidence,
    DurableResearchProgress,
)


class _Runner:
    def __init__(self) -> None:
        self.candidate_ids: list[str] = []
        self.analysis_ids: list[str] = []

    def run_candidate(self, candidate_id: str):
        self.candidate_ids.append(candidate_id)
        return {
            "candidate_id": candidate_id,
            "plan_id": "PLAN-AUTO-1",
            "media_count": 20,
            "claim_count": 0,
        }

    def run_analysis(self, analysis_id: str):
        self.analysis_ids.append(analysis_id)
        return {
            "analysis_id": analysis_id,
            "plan_id": "PLAN-AUTO-DAY-1",
            "media_count": 7,
            "claim_count": 4,
        }


def _incident_durable(*, completed: bool, documentary_ready: bool) -> DurableEventEvidence:
    event = EventEvidenceV1.model_validate(
        {
            "event_id": "AN-DIE-2026-07-06",
            "time_window": {"from_at": "2026-07-06T00:00:00Z"},
        }
    )
    progress = (
        DurableResearchProgress(
            plan_id="PLAN-ONGOING",
            plan_revision="1" * 64,
            wave_number=1,
            wave_focus=("incident_identity",),
            page_count=5,
            completed=False,
            media_ticket_limit=2_048,
            safety_limit_reached=False,
            converged=False,
            zero_yield_wave_streak=0,
            coverage_ready=False,
            next_cursor="cursor",
        )
        if completed
        else None
    )
    coverage = BackendIncidentDayCoverage(
        queries_exhausted=documentary_ready,
        safety_limit_reached=False,
        converged=documentary_ready,
        source_count=4 if documentary_ready else 0,
        official_source_count=1 if documentary_ready else 0,
        independent_evidence_family_count=3 if documentary_ready else 0,
        claim_count=4 if documentary_ready else 0,
        image_count=2 if documentary_ready else 0,
        video_count=1 if documentary_ready else 0,
        audio_count=0,
        media_analysis_required_count=1 if documentary_ready else 0,
        media_analysis_completed_count=0,
        media_analysis_failed_count=0,
        satellite_artifact_count=0,
        materialized_satellite_count=0,
        satellite_analysis_required_count=0,
        satellite_analysis_completed_count=0,
        spatial_observation_count=0,
        time_qualified_observation_count=2 if documentary_ready else 0,
        expected_lifecycle_phases=("daily_progression_or_status",),
        covered_lifecycle_phases=(("daily_progression_or_status",) if documentary_ready else ()),
        missing_dimensions=(
            ("spatial_observation", "public_media_analysis")
            if documentary_ready
            else ("web_query_waves", "spatial_observation")
        ),
        documentary_ready=documentary_ready,
        spatial_ready=False,
        satellite_analysis_ready=True,
        media_analysis_ready=not documentary_ready,
        coverage_ready=False,
    )
    return DurableEventEvidence(
        event=event,
        media_locations=(),
        vision_artifacts=(),
        upload_locations=(),
        prior_fire_states=(),
        geospatial_checks=(),
        geographic_references=(),
        source_revision_sha256="a" * 64,
        research_progress=progress,
        incident_id="FR-26-00001",
        viewpoint_label="Die Justin",
        research_target_kind="incident_day",
        incident_day_coverage=coverage,
    )


class _SequencedRepository:
    def __init__(self) -> None:
        initial = _incident_durable(completed=False, documentary_ready=False)
        ongoing = replace(
            _incident_durable(completed=True, documentary_ready=False),
            source_revision_sha256="b" * 64,
        )
        ready = replace(
            _incident_durable(completed=True, documentary_ready=True),
            source_revision_sha256="c" * 64,
        )
        self._values = [initial, ongoing, ready]

    def read(self, _event_id: str) -> DurableEventEvidence:
        return self._values.pop(0)


class _SequencedWorker:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, plan):
        self.calls += 1
        done = self.calls == 2
        return SourceAcquisitionRunReceipt(
            candidate_id=plan.candidate_id,
            plan_id=plan.plan_id,
            plan_revision=plan.revision,
            wave_number=plan.wave_number,
            wave_focus=plan.wave_focus,
            pages_published=5,
            source_count=4 if done else 1,
            claim_count=4 if done else 0,
            media_count=3 if done else 1,
            completed=done,
            media_ticket_limit=plan.media_ticket_limit,
            safety_limit_reached=False,
            converged=done,
            zero_yield_wave_streak=2 if done else 0,
            coverage_ready=done,
            next_cursor=None if done else "cursor",
            source_revision_sha256=("c" if done else "b") * 64,
        )


def _free_port() -> int:
    with socket.socket() as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def test_source_service_accepts_only_candidate_id_and_builds_no_manual_plan() -> None:
    runner = _Runner()
    token = "worker-" + ("w" * 40)
    settings = SourceAcquisitionServiceSettings(
        host="127.0.0.1",
        port=_free_port(),
        worker_token=token,
        broker_control_token="broker-" + ("b" * 40),
        managed_identity_client_id="22222222-2222-4222-8222-222222222222",
        aws_role_arn="arn:aws:iam::123456789012:role/fireviewer-source-bedrock",
        aws_oidc_audience="api://33333333-3333-4333-8333-333333333333",
        backend=AzureBackendEventEvidenceConfig(
            base_url="https://backend.fireviewer.test",
            bearer_token="backend-" + ("x" * 40),
        ),
    )
    service = SourceAcquisitionService(settings=settings, runner=runner)
    assert settings.max_pages_per_run == 1
    assert settings.results_per_page == 1
    assert settings.media_per_source == 2
    assert settings.max_multimodal_analyses_per_run == 1
    server = create_source_acquisition_server(service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    body = json.dumps({"candidate_id": "EC-SERVICE-1"}).encode()
    try:
        connection = HTTPConnection("127.0.0.1", settings.port, timeout=5)
        connection.request("GET", "/healthz")
        health = connection.getresponse()
        health_payload = json.loads(health.read())
        assert health.status == 200
        assert health_payload["runtime"] == "azure-cpu"
        assert health_payload["manual_plan_accepted"] is False
        assert health_payload["incident_day_supported"] is True
        assert health_payload["multimodal_provider"] == "aws-bedrock-pixtral"
        assert health_payload["multimodal_enabled"] is False
        assert health_payload["raw_scraped_content_stored"] is False

        connection.request(
            "POST",
            "/v1/event-evidence/research",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        denied = connection.getresponse()
        denied.read()
        assert denied.status == 401

        connection.request(
            "POST",
            "/v1/event-evidence/research",
            body=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        accepted = connection.getresponse()
        accepted_payload = json.loads(accepted.read())
        assert accepted.status == 200
        assert accepted_payload["media_count"] == 20
        assert accepted_payload["claim_count"] == 0
        assert runner.candidate_ids == ["EC-SERVICE-1"]

        analysis_body = json.dumps({"analysis_id": "AN-DIE-2026-07-06"}).encode()
        connection.request(
            "POST",
            "/v1/incident-day/research",
            body=analysis_body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        analysis_response = connection.getresponse()
        analysis_payload = json.loads(analysis_response.read())
        assert analysis_response.status == 200
        assert analysis_payload["analysis_id"] == "AN-DIE-2026-07-06"
        assert analysis_payload["media_count"] == 7
        assert runner.analysis_ids == ["AN-DIE-2026-07-06"]

        oversized_plan = json.dumps(
            {
                "candidate_id": "EC-SERVICE-1",
                "queries": ["manual plan is forbidden"],
            }
        ).encode()
        connection.request(
            "POST",
            "/v1/event-evidence/research",
            body=oversized_plan,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        rejected = connection.getresponse()
        rejected.read()
        assert rejected.status == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_incident_orchestrator_resumes_pages_until_documentary_coverage() -> None:
    repository = _SequencedRepository()
    worker = _SequencedWorker()
    orchestrator = SourceAcquisitionOrchestrator(
        repository=repository,
        incident_repository=repository,
        planner=AutomaticSourceAcquisitionPlanner(),
        worker=worker,
        incident_worker=worker,
        max_incident_cycles=2,
    )

    result = orchestrator.run_analysis("AN-DIE-2026-07-06")

    assert worker.calls == 2
    assert result["analysis_id"] == "AN-DIE-2026-07-06"
    assert result["orchestration_cycles"] == 2
    assert result["orchestration_complete"] is True
    assert result["daily_bundle_ready"] is False
    assert result["coverage_ready"] is True
    assert result["missing_dimensions"] == [
        "spatial_observation",
        "public_media_analysis",
    ]
