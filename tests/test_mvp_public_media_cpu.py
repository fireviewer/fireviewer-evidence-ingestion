from __future__ import annotations

import socket
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from PIL import Image
from pydantic import SecretStr

from fireviewer_contracts.mvp.contracts import (
    Detection,
    DetectionResultV1,
    EventEvidenceV1,
    EvidenceMedia,
    EvidenceSource,
    ProviderRun,
)
from fireviewer_evidence_ingestion.mvp.research.multimodal_evidence import (
    ExtractedMultimodalClaim,
    MultimodalEvidenceExtraction,
)
from fireviewer_evidence_ingestion.mvp.research.public_media_cpu import (
    PROCESSOR_ID,
    PROCESSOR_REVISION,
    HttpTransientYoloDetector,
    PublicMediaCpuConfig,
    PublicMediaCpuWorker,
)
from fireviewer_evidence_ingestion.mvp.research.transcription import TransientTranscript
from fireviewer_contracts.mvp.supervision.backend_event_evidence import (
    BackendResearchMedia,
    BackendResearchMediaAnalysisBatch,
    BackendResearchMediaAnalysisReceipt,
    DurableEventEvidence,
    DurableResearchProgress,
)
from fireviewer_vision_runtime.mvp.vision.video_keyframes import (
    VideoKeyframeArtifact,
    VideoKeyframeExtractor,
    VideoKeyframeRun,
)
from fireviewer_evidence_ingestion.research_broker import ResearchBroker


def _png(color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (16, 12), color=color).save(output, format="PNG")
    return output.getvalue()


def _durable(video_bytes: bytes, *, processed: bool = False) -> DurableEventEvidence:
    source = EvidenceSource(
        source_id="SOURCE-PUBLIC-1",
        origin_id="ORIGIN-PUBLIC-1",
        source_url="https://sources.example/incident-update",
        publisher="Public authority",
        published_at=datetime(2026, 7, 6, 12, tzinfo=UTC),
        retrieved_at=datetime(2026, 7, 6, 13, tzinfo=UTC),
        source_type="official",
        independence_weight=1,
    )
    media = BackendResearchMedia(
        media_id="MEDIA-PUBLIC-VIDEO-1",
        source_id=source.source_id,
        media_group_id="GROUP-PUBLIC-1",
        origin_id="ORIGIN-PUBLIC-VIDEO-1",
        kind="video",
        sha256=sha256(video_bytes).hexdigest(),
        captured_at=datetime(2026, 7, 6, 12, tzinfo=UTC),
        source_url="https://sources.example/point-press.mp4",
        content_type="video/mp4",
        size_bytes=len(video_bytes),
    )
    batches = (
        (
            BackendResearchMediaAnalysisBatch(
                batch_id="MEDIA-BATCH-EXISTING",
                media_id=media.media_id,
                media_sha256=media.sha256,
                processor_id=PROCESSOR_ID,
                processor_revision=PROCESSOR_REVISION,
                analyzed_at=datetime(2026, 7, 6, 14, tzinfo=UTC),
                outcome="success",
                request_sha256="f" * 64,
                claim_count=1,
                keyframe_observation_count=1,
                transcription_receipt_count=1,
                journal_entry_count=4,
                raw_content_stored=False,
            ),
        )
        if processed
        else ()
    )
    return DurableEventEvidence(
        event=EventEvidenceV1(
            event_id="AN-DIE-2026-07-06",
            sources=(source,),
            media=(media,),
            needs_human_review=True,
        ),
        media_locations=(),
        vision_artifacts=(),
        upload_locations=(),
        prior_fire_states=(),
        geospatial_checks=(),
        geographic_references=(),
        source_revision_sha256="a" * 64,
        research_progress=DurableResearchProgress(
            plan_id="PLAN-PUBLIC-1",
            plan_revision="9" * 64,
            wave_number=2,
            wave_focus=("daily_progression",),
            page_count=4,
            completed=True,
            media_ticket_limit=2_048,
            safety_limit_reached=False,
            converged=True,
            zero_yield_wave_streak=2,
            coverage_ready=True,
            next_cursor=None,
        ),
        research_source_policies={
            "sources.example": {
                "publisher": "Public authority",
                "source_type": "official",
                "independence_weight": 1,
                "claim_types": ["incident_status", "fire_progression"],
            }
        },
        research_search_templates={"search.example": "https://search.example/search?q={query}"},
        research_target_kind="incident_day",
        research_media_tickets=(media,),
        research_media_analysis_batches=batches,
    )


def _photo_durable(
    photo_bytes: bytes,
    *,
    batches: tuple[BackendResearchMediaAnalysisBatch, ...] = (),
) -> DurableEventEvidence:
    source = EvidenceSource(
        source_id="SOURCE-PUBLIC-PHOTO-1",
        origin_id="ORIGIN-PUBLIC-PHOTO-1",
        source_url="https://sources.example/incident-gallery",
        publisher="Public authority",
        published_at=datetime(2026, 7, 6, 12, tzinfo=UTC),
        retrieved_at=datetime(2026, 7, 6, 13, tzinfo=UTC),
        source_type="official",
        independence_weight=1,
    )
    media = BackendResearchMedia(
        media_id="MEDIA-PUBLIC-PHOTO-1",
        source_id=source.source_id,
        media_group_id="GROUP-PUBLIC-PHOTO-1",
        origin_id="ORIGIN-PUBLIC-PHOTO-1",
        kind="photo",
        sha256=sha256(photo_bytes).hexdigest(),
        captured_at=datetime(2026, 7, 6, 12, tzinfo=UTC),
        source_url="https://sources.example/front.jpg",
        content_type="image/png",
        size_bytes=len(photo_bytes),
    )
    return DurableEventEvidence(
        event=EventEvidenceV1(
            event_id="AN-DIE-2026-07-06-PHOTO",
            sources=(source,),
            media=(media,),
            needs_human_review=True,
        ),
        media_locations=(),
        vision_artifacts=(),
        upload_locations=(),
        prior_fire_states=(),
        geospatial_checks=(),
        geographic_references=(),
        source_revision_sha256="a" * 64,
        research_progress=DurableResearchProgress(
            plan_id="PLAN-PUBLIC-PHOTO-1",
            plan_revision="9" * 64,
            wave_number=2,
            wave_focus=("visual_evidence",),
            page_count=4,
            completed=True,
            media_ticket_limit=2_048,
            safety_limit_reached=False,
            converged=True,
            zero_yield_wave_streak=2,
            coverage_ready=True,
            next_cursor=None,
        ),
        research_source_policies={
            "sources.example": {
                "publisher": "Public authority",
                "source_type": "official",
                "independence_weight": 1,
                "claim_types": ["incident_status", "fire_progression"],
            }
        },
        research_search_templates={"search.example": "https://search.example/search?q={query}"},
        research_target_kind="incident_day",
        research_media_tickets=(media,),
        research_media_analysis_batches=batches,
    )


class _Repository:
    def __init__(self, durable: DurableEventEvidence) -> None:
        self.durable = durable

    def read(self, event_id: str) -> DurableEventEvidence:
        assert event_id == self.durable.event.event_id
        return self.durable


class _Publisher:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def publish(self, *, candidate_id: str, payload) -> BackendResearchMediaAnalysisReceipt:
        self.payloads.append(dict(payload))
        return BackendResearchMediaAnalysisReceipt(
            candidate_id=candidate_id,
            batch_id=str(payload["batch_id"]),
            media_id=str(payload["media_id"]),
            replayed=False,
            claim_count=len(payload["claims"]),
            keyframe_observation_count=len(payload["keyframe_observations"]),
            transcription_receipt_count=len(payload["transcription_receipts"]),
            journal_entry_count=len(payload["journal_entries"]),
            source_revision_sha256="b" * 64,
        )


class _Detector:
    def detect(self, *, media_id: str, content_type: str, content: bytes) -> DetectionResultV1:
        assert content_type == "image/png"
        return DetectionResultV1(
            media_id=media_id,
            provider_run=ProviderRun(
                provider_id="yolo-fire-smoke-cpu",
                provider_version="1.0.0",
                model_id="fire-smoke-yolo",
                model_version="immutable-yolo-revision",
                config={"device": "cpu"},
                input_hash=sha256(content).hexdigest(),
                runtime_ms=10,
                cost_usd=0,
                generated_at=datetime(2026, 7, 6, 14, tzinfo=UTC),
            ),
            detections=(
                Detection(
                    detection_id=f"DET-{media_id}",
                    detection_class="smoke",
                    bbox=(0.1, 0.2, 0.8, 0.9),
                    score=0.9,
                    prompt="smoke",
                ),
            ),
            status="smoke",
            needs_human_review=True,
        )


class _AudioExtractor:
    def __init__(self) -> None:
        self.source_path: Path | None = None

    def extract(self, source: Path, destination: Path) -> Path:
        self.source_path = source
        destination.write_bytes(b"RIFF" + (b"0" * 64))
        return destination


class _Transcriber:
    provider_id = "managed-transcription-test"
    model_revision = "managed-transcription-v1"

    def transcribe(self, path: Path, *, content_type: str) -> TransientTranscript:
        assert path.is_file()
        assert content_type == "audio/wav"
        text = "Le front progresse vers le nord selon le point presse officiel."
        return TransientTranscript(
            provider_id=self.provider_id,
            model_revision=self.model_revision,
            text=text,
            transcript_sha256=sha256(text.encode()).hexdigest(),
            duration_seconds=12,
            language="fr-FR",
            confidence=0.91,
            partial=False,
        )


class _EvidenceProvider:
    provider_id = "managed-vl-test"

    def extract(self, document, *, allowed_claim_types) -> MultimodalEvidenceExtraction:
        assert document.content_role == "transcript"
        assert len(document.images) == 2
        assert "fire_progression" in allowed_claim_types
        return MultimodalEvidenceExtraction(
            provider_id=self.provider_id,
            model_revision="managed-vl-v1",
            prompt_revision="8" * 64,
            claims=(
                ExtractedMultimodalClaim(
                    claim_type="fire_progression",
                    text="Le front progresse vers le nord.",
                    observed_at=datetime(2026, 7, 6, 12, tzinfo=UTC),
                    confidence=0.9,
                    evidence_media_ids=(document.images[0].media_id,),
                ),
            ),
            partial=False,
        )


class _PhotoEvidenceProvider:
    provider_id = "managed-vl-test"

    def extract(self, document, *, allowed_claim_types) -> MultimodalEvidenceExtraction:
        assert document.content_role == "page"
        assert len(document.images) == 1
        assert document.images[0].content_type == "image/png"
        assert "fire_progression" in allowed_claim_types
        return MultimodalEvidenceExtraction(
            provider_id=self.provider_id,
            model_revision="managed-vl-v1",
            prompt_revision="8" * 64,
            claims=(
                ExtractedMultimodalClaim(
                    claim_type="fire_progression",
                    text="Une fumée est visible sur le versant.",
                    observed_at=datetime(2026, 7, 6, 12, tzinfo=UTC),
                    confidence=0.88,
                    evidence_media_ids=(document.images[0].media_id,),
                ),
            ),
            partial=False,
        )


def _public_dns(*_args, **_kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]


def test_transient_yolo_rejects_an_oversized_response() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=b"{" + (b" " * (1_024 * 1_024)),
            request=request,
        )
    )
    with httpx.Client(transport=transport) as client:
        detector = HttpTransientYoloDetector(
            endpoint="https://yolo.internal.example",
            token=SecretStr("t" * 32),
            client=client,
        )
        with pytest.raises(RuntimeError, match="response exceeds its byte limit"):
            detector.detect(
                media_id="MEDIA-1",
                content_type="image/png",
                content=_png((255, 80, 20)),
            )


def test_public_video_runs_keyframes_yolo_transcription_and_ticket_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    video = b"bounded-public-video"
    frame_data = (_png((255, 80, 20)), _png((100, 100, 100)))
    durable = _durable(video)
    parent = durable.research_media_tickets[0]
    artifacts = tuple(
        VideoKeyframeArtifact(
            media=EvidenceMedia(
                media_id=f"KF-PUBLIC-{index}",
                source_id=parent.source_id,
                media_group_id=parent.media_group_id,
                origin_id=parent.origin_id,
                kind="keyframe",
                sha256=sha256(content).hexdigest(),
                parent_media_id=parent.media_id,
            ),
            frame_index=index * 30,
            timestamp_seconds=float(index),
            format="png",
            data=content,
        )
        for index, content in enumerate(frame_data, start=1)
    )
    monkeypatch.setattr(
        VideoKeyframeExtractor,
        "run",
        lambda _self, event: VideoKeyframeRun(evidence=event, artifacts=artifacts),
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={
                "content-type": "video/mp4",
                "content-length": str(len(video)),
            },
            content=video,
            request=request,
        )
    )
    publisher = _Publisher()
    audio = _AudioExtractor()
    worker = PublicMediaCpuWorker(
        repository=_Repository(durable),
        publisher=publisher,
        broker=ResearchBroker(control_token="c" * 40, transport=transport),
        broker_control_token="c" * 40,
        detector=_Detector(),
        transcription_provider=_Transcriber(),
        evidence_provider=_EvidenceProvider(),
        audio_extractor=audio,
        clock=lambda: datetime(2026, 7, 6, 14, tzinfo=UTC),
    )

    receipt = worker.run_analysis(durable.event.event_id)

    assert receipt.succeeded_count == 1
    assert receipt.failed_count == 0
    assert receipt.source_revision_sha256 == "b" * 64
    assert receipt.raw_public_media_stored is False
    payload = publisher.payloads[0]
    assert payload["outcome"] == "success"
    assert len(payload["keyframe_observations"]) == 2
    assert payload["keyframe_observations"][0]["media_id"] == parent.media_id
    assert payload["keyframe_observations"][0]["frame_binary_stored"] is False
    assert payload["transcription_receipts"][0]["transcript_stored"] is False
    assert payload["claims"][0]["evidence_media_ids"] == [parent.media_id]
    assert payload["source_binary_stored"] is False
    assert audio.source_path is not None and not audio.source_path.exists()
    assert video.decode() not in str(payload)


def test_public_media_worker_skips_same_successful_processor_revision() -> None:
    video = b"already-processed-public-video"
    durable = _durable(video, processed=True)
    publisher = _Publisher()
    worker = PublicMediaCpuWorker(
        repository=_Repository(durable),
        publisher=publisher,
        broker=ResearchBroker(control_token="c" * 40),
        broker_control_token="c" * 40,
        detector=_Detector(),
        transcription_provider=_Transcriber(),
        evidence_provider=_EvidenceProvider(),
        audio_extractor=_AudioExtractor(),
    )

    receipt = worker.run_analysis(durable.event.event_id)

    assert receipt.already_processed_count == 1
    assert receipt.attempted_count == 0
    assert receipt.remaining_count == 0
    assert publisher.payloads == []


def test_public_photo_runs_transient_yolo_and_vl_without_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    photo = _png((255, 80, 20))
    durable = _photo_durable(photo)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={
                "content-type": "image/png",
                "content-length": str(len(photo)),
            },
            content=photo,
            request=request,
        )
    )
    publisher = _Publisher()
    worker = PublicMediaCpuWorker(
        repository=_Repository(durable),
        publisher=publisher,
        broker=ResearchBroker(control_token="c" * 40, transport=transport),
        broker_control_token="c" * 40,
        detector=_Detector(),
        transcription_provider=None,
        evidence_provider=_PhotoEvidenceProvider(),
        audio_extractor=_AudioExtractor(),
        clock=lambda: datetime(2026, 7, 6, 14, tzinfo=UTC),
    )

    receipt = worker.run_analysis(durable.event.event_id)

    assert receipt.succeeded_count == 1
    assert receipt.remaining_count == 0
    payload = publisher.payloads[0]
    assert payload["outcome"] == "success"
    assert len(payload["keyframe_observations"]) == 1
    assert payload["keyframe_observations"][0]["media_id"] == "MEDIA-PUBLIC-PHOTO-1"
    assert payload["keyframe_observations"][0]["frame_binary_stored"] is False
    assert payload["transcription_receipts"] == []
    assert payload["source_binary_stored"] is False
    assert photo.hex() not in str(payload)


def test_exhausted_partial_media_does_not_block_the_durable_stage() -> None:
    photo = _png((40, 50, 60))
    digest = sha256(photo).hexdigest()
    batches = tuple(
        BackendResearchMediaAnalysisBatch(
            batch_id=f"MEDIA-BATCH-PARTIAL-{attempt}",
            media_id="MEDIA-PUBLIC-PHOTO-1",
            media_sha256=digest,
            processor_id=PROCESSOR_ID,
            processor_revision=PROCESSOR_REVISION,
            analyzed_at=datetime(2026, 7, 6, 14 + attempt, tzinfo=UTC),
            outcome="partial",
            request_sha256=f"{attempt}" * 64,
            claim_count=1,
            keyframe_observation_count=1,
            transcription_receipt_count=0,
            journal_entry_count=3,
            raw_content_stored=False,
        )
        for attempt in range(1, 4)
    )
    durable = _photo_durable(photo, batches=batches)
    publisher = _Publisher()
    worker = PublicMediaCpuWorker(
        repository=_Repository(durable),
        publisher=publisher,
        broker=ResearchBroker(control_token="c" * 40),
        broker_control_token="c" * 40,
        detector=_Detector(),
        transcription_provider=None,
        evidence_provider=_PhotoEvidenceProvider(),
        audio_extractor=_AudioExtractor(),
        config=PublicMediaCpuConfig(maximum_attempts_per_media=3),
    )

    receipt = worker.run_analysis(durable.event.event_id)

    assert receipt.attempted_count == 0
    assert receipt.remaining_count == 0
    assert publisher.payloads == []
