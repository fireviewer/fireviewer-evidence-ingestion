from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta

import fireviewer_evidence_ingestion.research_service as research_service
from fireviewer_evidence_ingestion.research_client import ResearchServiceError, run_isolated_research


def _input() -> dict[str, object]:
    now = datetime.now(UTC)
    cutoff = now + timedelta(hours=1)
    return {
        "schema_version": "research-1.0",
        "operation": "source_research",
        "research_id": "research-synthetic-0001",
        "analysis_window": {
            "analysis_id": "analysis-synthetic-0001",
            "fire_id": "FR-99-00001",
            "episode_id": "E01",
            "window_start_at": now.isoformat(),
            "window_end_at": cutoff.isoformat(),
            "local_date": now.date().isoformat(),
            "timezone": "Europe/Paris",
        },
        "incident_name": "Incident synthétique",
        "incident_reference": [5.37, 44.75],
        "cutoff_at": cutoff.isoformat(),
        "location_hint": "Commune fictive, secteur de démonstration",
        "source_registry_version": "firewarning-fr-sources-2026-07-19-v1",
        "allowed_domains": ["authority.example"],
        "source_policies": {
            "authority.example": {
                "source_name": "Autorité de démonstration",
                "kind": "authority",
                "scope": "local",
                "confidence_level": "A+",
                "claim_types": ["operational_confirmation", "local_instruction"],
                "publication_policy": "per_item_license_check",
                "minimum_refresh_minutes": 10,
            }
        },
        "search_templates": {"search.example": "https://search.example/recherche?q={query}"},
        "max_fetch_bytes": 1_048_576,
        "request_timeout_seconds": 20,
        "private_upload": {
            "pathname_prefix": "firewarning/source-packages/upload-test",
            "upload_grant": "g" * 128,
            "token_endpoint": "https://fireviewer.example/api/v1/admin/blob-upload-token",
            "resource_id": "research-synthetic-0001",
            "maximum_file_size_bytes": 10_485_760,
            "allowed_content_types": ["image/jpeg", "text/html"],
        },
    }


def test_service_keeps_upload_grant_out_of_qwen_process(monkeypatch) -> None:
    now = datetime.now(UTC)
    broker_calls: list[dict[str, object]] = []
    child_inputs: list[dict[str, object]] = []
    monkeypatch.setenv(
        "FW_RESEARCH_BROKER_CONTROL_TOKEN",
        "control-credential-for-service-tests-000000000000",
    )

    def fake_broker(value):
        broker_calls.append(value)
        if value["action"] == "configure":
            return {"session_token": "session-token-for-tests-000000000000000000"}
        return {"revoked": True}

    def fake_run(*_args, **kwargs):
        child_payload = json.loads(kwargs["input"])
        child_inputs.append(child_payload)
        assert "private_upload" not in child_payload
        assert "upload_grant" not in kwargs["env"]
        output = {
            "schema_version": "research-1.0",
            "research_id": "research-synthetic-0001",
            "status": "succeeded",
            "retryable": False,
            "model_run": {
                "model_id": "Qwen/Qwen3-14B",
                "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
                "status": "succeeded",
                "started_at": now.isoformat(),
                "finished_at": now.isoformat(),
                "load_ms": 1,
                "inference_ms": 1,
            },
            "queries": [],
            "candidates": [],
            "validation_errors": [],
        }
        return subprocess.CompletedProcess([], 0, stdout=json.dumps(output), stderr="")

    monkeypatch.setattr(research_service, "_broker_call", fake_broker)
    monkeypatch.setattr(research_service.subprocess, "run", fake_run)

    output = research_service.execute_research(_input())

    assert output["status"] == "succeeded"
    assert len(child_inputs) == 1
    assert [call["action"] for call in broker_calls] == ["configure", "revoke"]
    assert broker_calls[0]["policy"]["upload_grant"] == "g" * 128


def test_service_preserves_sandbox_failure_detail(monkeypatch) -> None:
    monkeypatch.setenv(
        "FW_RESEARCH_BROKER_CONTROL_TOKEN",
        "control-credential-for-service-tests-000000000000",
    )

    def fake_broker(value):
        if value["action"] == "configure":
            return {"session_token": "session-token-for-tests-000000000000000000"}
        return {"revoked": True}

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 1, stdout="", stderr="model load failed")

    monkeypatch.setattr(research_service, "_broker_call", fake_broker)
    monkeypatch.setattr(research_service.subprocess, "run", fake_run)

    try:
        research_service.execute_research(_input())
    except research_service.ResearchServiceError as exc:
        assert exc.code == "sandboxed_research_process_failed"
        assert exc.detail == "model load failed"
    else:
        raise AssertionError("expected the sandboxed process failure")


def test_client_preserves_structured_service_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "fireviewer_evidence_ingestion.research_client.call",
        lambda *_args, **_kwargs: {
            "ok": False,
            "error": {
                "code": "sandboxed_research_process_failed",
                "detail": "model load failed",
            },
        },
    )

    try:
        run_isolated_research(research_service.ResearchInputV1.model_validate(_input()))
    except ResearchServiceError as exc:
        assert exc.code == "sandboxed_research_process_failed"
        assert exc.detail == "model load failed"
    else:
        raise AssertionError("expected the structured service failure")
