from __future__ import annotations

import socket
from datetime import UTC, datetime
from typing import Any

import httpx

from fireviewer_contracts.mvp.contracts import EventEvidenceV1
from fireviewer_evidence_ingestion.mvp.research.multimodal_evidence import (
    ExtractedMultimodalClaim,
    MultimodalEvidenceDocument,
    MultimodalEvidenceExtraction,
)
from fireviewer_evidence_ingestion.mvp.research.source_acquisition import (
    CpuSourceAcquisitionWorker,
    SourceAcquisitionPlan,
)
from fireviewer_contracts.mvp.supervision.backend_event_evidence import (
    BackendResearchEvidenceReceipt,
    DurableEventEvidence,
)
from fireviewer_evidence_ingestion.research_broker import ResearchBroker

NOW = datetime(2026, 8, 23, 12, tzinfo=UTC)
CONTROL_TOKEN = "source-worker-control-token-000000000000000000"  # noqa: S105


class _MultimodalProvider:
    provider_id = "aws-bedrock-pixtral"

    def __init__(self) -> None:
        self.documents: list[MultimodalEvidenceDocument] = []

    def extract(self, document, *, allowed_claim_types):
        self.documents.append(document)
        assert "area_burned" in allowed_claim_types
        return MultimodalEvidenceExtraction(
            provider_id=self.provider_id,
            model_revision="mistral.pixtral-large-2502-v1:0",
            prompt_revision="f" * 64,
            claims=(
                ExtractedMultimodalClaim(
                    claim_type="area_burned",
                    text="La source rapporte 120 hectares parcourus.",
                    confidence=0.92,
                    evidence_media_ids=(document.images[0].media_id,),
                ),
            ),
        )


class _Repository:
    def __init__(self, event_id: str) -> None:
        self.value = DurableEventEvidence(
            event=EventEvidenceV1(event_id=event_id),
            media_locations=(),
            vision_artifacts=(),
            upload_locations=(),
            prior_fire_states=(),
            geospatial_checks=(),
            geographic_references=(),
            source_revision_sha256="0" * 64,
        )

    def read(self, event_id: str) -> DurableEventEvidence:
        assert event_id == self.value.event.event_id
        return self.value


class _Publisher:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.sources: set[str] = set()
        self.claims: set[str] = set()
        self.media: set[str] = set()

    def publish(
        self,
        *,
        candidate_id: str,
        payload: dict[str, Any],
    ) -> BackendResearchEvidenceReceipt:
        self.payloads.append(payload)
        self.sources.update(item["source_id"] for item in payload["sources"])
        self.claims.update(item["claim_id"] for item in payload["claims"])
        self.media.update(item["media_id"] for item in payload["media"])
        revision = f"{len(self.payloads):064x}"
        return BackendResearchEvidenceReceipt(
            candidate_id=candidate_id,
            plan_id=payload["plan_id"],
            page_id=payload["page_id"],
            wave_number=payload["wave_number"],
            wave_focus=tuple(payload["wave_focus"]),
            replayed=False,
            source_count=len(self.sources),
            claim_count=len(self.claims),
            media_count=len(self.media),
            duplicate_source_count=0,
            duplicate_claim_count=0,
            duplicate_media_count=0,
            completed=payload["completed"],
            media_ticket_limit=payload["media_ticket_limit"],
            safety_limit_reached=payload["safety_limit_reached"],
            converged=payload["converged"],
            zero_yield_wave_streak=payload["zero_yield_wave_streak"],
            coverage_ready=payload["coverage_ready"],
            next_cursor=payload["next_cursor"],
            source_revision_sha256=revision,
        )


def _public_dns(*_args, **_kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]


def _plan() -> SourceAcquisitionPlan:
    return SourceAcquisitionPlan(
        candidate_id="EC-SOURCE-TEST-1",
        plan_id="PLAN-SOURCE-TEST-1",
        wave_focus=("incident_identity", "collection_convergence"),
        queries=("incendie test", "bilan test", "dernieres informations test"),
        allowed_domains=("sources.example",),
        source_policies={
            "sources.example": {
                "publisher": "Service officiel",
                "source_type": "official",
                "independence_weight": 0.95,
            }
        },
        search_provider_domain="search.example",
        search_template="https://search.example/search?q={query}",
        media_ticket_limit=100,
        results_per_page=2,
        max_pages_per_run=12,
    )


def test_cpu_worker_collects_until_query_convergence_without_a_media_target(
    monkeypatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "search.example":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=b"""
                <a href="https://sources.example/article-1">one</a>
                <a href="https://sources.example/article-2">two</a>
                <a href="https://sources.example/article-3">three</a>
                """,
                request=request,
            )
        if request.url.path.startswith("/article-"):
            index = request.url.path.rsplit("-", 1)[1]
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=(
                    '<meta property="og:site_name" content="Official source">'
                    '<meta property="og:description" content="RAW ARTICLE MUST NOT PERSIST">'
                    f'<meta property="og:image" content="/media-{index}.jpg">'
                ).encode(),
                request=request,
            )
        return httpx.Response(
            200,
            headers={"content-type": "image/jpeg"},
            content=(b"\xff\xd8\xff" + request.url.path.encode()),
            request=request,
        )

    broker = ResearchBroker(
        control_token=CONTROL_TOKEN,
        transport=httpx.MockTransport(handler),
    )
    publisher = _Publisher()
    worker = CpuSourceAcquisitionWorker(
        repository=_Repository("EC-SOURCE-TEST-1"),
        publisher=publisher,
        broker=broker,
        broker_control_token=CONTROL_TOKEN,
        clock=lambda: NOW,
    )

    receipt = worker.run(_plan())

    assert receipt.pages_published >= 4
    assert receipt.completed is True
    assert receipt.converged is True
    assert receipt.coverage_ready is True
    assert receipt.source_count == 3
    assert receipt.claim_count == 0
    assert receipt.media_count == 3
    assert publisher.payloads[0]["next_cursor"] is not None
    assert publisher.payloads[-1]["next_cursor"] is None
    serialized = str(publisher.payloads)
    assert "RAW ARTICLE MUST NOT PERSIST" not in serialized
    assert all(payload["claims"] == [] for payload in publisher.payloads)
    assert all("content" not in item for payload in publisher.payloads for item in payload["media"])


def test_cpu_worker_persists_fetch_failures_as_structured_journal(monkeypatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "search.example":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=b'<a href="https://sources.example/article-1">one</a>',
                request=request,
            )
        return httpx.Response(503, request=request)

    publisher = _Publisher()
    worker = CpuSourceAcquisitionWorker(
        repository=_Repository("EC-SOURCE-TEST-1"),
        publisher=publisher,
        broker=ResearchBroker(
            control_token=CONTROL_TOKEN,
            transport=httpx.MockTransport(handler),
        ),
        broker_control_token=CONTROL_TOKEN,
        clock=lambda: NOW,
    )

    receipt = worker.run(
        _plan().model_copy(
            update={
                "queries": ("incendie test",),
                "media_ticket_limit": 100,
            }
        )
    )

    assert receipt.completed is True
    journal = publisher.payloads[0]["journal_entries"]
    assert journal[0]["stage"] == "page_fetch"
    assert journal[0]["outcome"] == "failed"
    assert journal[0]["source_url"] == "https://sources.example/article-1"
    assert "raw article must not persist" not in journal[0]["detail"].casefold()
    assert any(item["error_code"] == "collection_not_converged" for item in journal)
    assert publisher.payloads[0]["sources"] == []
    assert publisher.payloads[0]["claims"] == []
    assert publisher.payloads[0]["media"] == []


def test_cpu_worker_sends_public_page_and_image_transiently_to_mistral_vl(monkeypatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "search.example":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=b'<a href="https://sources.example/article-1">one</a>',
                request=request,
            )
        if request.url.path == "/article-1":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=(
                    b'<meta property="og:image" content="/fire.jpg">'
                    b"<p>RAW ARTICLE 120 hectares</p>"
                ),
                request=request,
            )
        return httpx.Response(
            200,
            headers={"content-type": "image/jpeg"},
            content=b"\xff\xd8\xffTRANSIENT PUBLIC IMAGE",
            request=request,
        )

    provider = _MultimodalProvider()
    publisher = _Publisher()
    worker = CpuSourceAcquisitionWorker(
        repository=_Repository("EC-SOURCE-TEST-1"),
        publisher=publisher,
        broker=ResearchBroker(
            control_token=CONTROL_TOKEN,
            transport=httpx.MockTransport(handler),
        ),
        broker_control_token=CONTROL_TOKEN,
        multimodal_evidence_provider=provider,
        clock=lambda: NOW,
    )

    receipt = worker.run(
        _plan().model_copy(
            update={"queries": ("incendie test",), "media_ticket_limit": 100}
        )
    )

    assert receipt.claim_count == 1
    assert provider.documents[0].images[0].content.endswith(b"TRANSIENT PUBLIC IMAGE")
    assert publisher.payloads[0]["claims"][0]["claim_type"] == "area_burned"
    assert publisher.payloads[0]["journal_entries"][0]["provider_id"] == provider.provider_id
    serialized = str(publisher.payloads)
    assert "RAW ARTICLE" not in serialized
    assert "TRANSIENT PUBLIC IMAGE" not in serialized
