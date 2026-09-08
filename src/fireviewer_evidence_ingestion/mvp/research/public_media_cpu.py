"""CPU processing of public video/audio with ticket-only durable output."""

from __future__ import annotations

import base64
import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

import httpx
from PIL import Image, ImageOps
from pydantic import AnyHttpUrl, Field, SecretStr

from fireviewer_contracts.contracts import SafeIdentifierV2, Sha256HexV2, StrictModel
from fireviewer_contracts.mvp.contracts import DetectionResultV1, EventEvidenceV1, EvidenceMedia
from fireviewer_evidence_ingestion.mvp.research.multimodal_evidence import (
    MultimodalEvidenceDocument,
    MultimodalEvidenceProvider,
    MultimodalEvidenceProviderError,
    TransientEvidenceImage,
)
from fireviewer_evidence_ingestion.mvp.research.transcription import (
    AudioTrackExtractor,
    TranscriptionProvider,
    TranscriptionProviderError,
    TransientTranscript,
)
from fireviewer_contracts.mvp.supervision.backend_event_evidence import (
    BackendResearchMedia,
    BackendResearchMediaAnalysisReceipt,
    DurableEventEvidence,
    EventEvidenceRepository,
)
from fireviewer_vision_runtime.mvp.vision.video_keyframes import (
    OpenCvVideoFrameDecoder,
    VideoKeyframeArtifact,
    VideoKeyframeConfig,
    VideoKeyframeExtractor,
)
from fireviewer_evidence_ingestion.research_broker import BrokerPolicy, ResearchBroker

PROCESSOR_ID = "fireviewer-public-media-cpu"
PROCESSOR_REVISION = "public-media-cpu-1.1.0"
_MAX_TRANSIENT_PNG_BYTES = 5 * 1_024 * 1_024
_MAX_YOLO_RESPONSE_BYTES = 1 * 1_024 * 1_024


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}-{sha256(value.encode()).hexdigest()[:24]}"


class PublicMediaAnalysisPublisher(Protocol):
    def publish(
        self,
        *,
        candidate_id: str,
        payload: Mapping[str, Any],
    ) -> BackendResearchMediaAnalysisReceipt: ...


class TransientFrameDetector(Protocol):
    def detect(
        self,
        *,
        media_id: str,
        content_type: Literal["image/png"],
        content: bytes,
    ) -> DetectionResultV1: ...


class HttpTransientYoloDetector:
    def __init__(
        self,
        *,
        endpoint: str,
        token: SecretStr,
        timeout_seconds: float = 120,
        client: httpx.Client | None = None,
    ) -> None:
        parsed = urlsplit(endpoint.rstrip("/"))
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("transient YOLO endpoint must be HTTPS")
        self._endpoint = endpoint.rstrip("/")
        self._token = token
        self._timeout_seconds = timeout_seconds
        self._client = client

    def detect(
        self,
        *,
        media_id: str,
        content_type: Literal["image/png"],
        content: bytes,
    ) -> DetectionResultV1:
        if not content or len(content) > 5 * 1_024 * 1_024:
            raise ValueError("transient YOLO frame exceeds its byte limit")
        owned_client = self._client is None
        client = self._client or httpx.Client(
            timeout=httpx.Timeout(self._timeout_seconds, connect=10),
            follow_redirects=False,
            trust_env=False,
        )
        try:
            with client.stream(
                "POST",
                self._endpoint + "/v1/transient-images/detect",
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self._token.get_secret_value()}",
                },
                json={
                    "media_id": media_id,
                    "content_type": content_type,
                    "content_sha256": sha256(content).hexdigest(),
                    "content_base64": base64.b64encode(content).decode("ascii"),
                },
            ) as response:
                if response.is_redirect:
                    raise RuntimeError("transient YOLO redirect is forbidden")
                response.raise_for_status()
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > _MAX_YOLO_RESPONSE_BYTES:
                        raise RuntimeError("transient YOLO response exceeds its byte limit")
            payload = json.loads(body)
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError("transient YOLO request failed") from exc
        finally:
            if owned_client:
                client.close()
        if not isinstance(payload, Mapping):
            raise RuntimeError("transient YOLO response is invalid")
        if (
            payload.get("input_binary_stored") is not False
            or payload.get("geographic_output_created") is not False
        ):
            raise RuntimeError("transient YOLO retention contract was violated")
        result = DetectionResultV1.model_validate(payload.get("result"))
        if result.media_id != media_id:
            raise RuntimeError("transient YOLO media identity mismatch")
        return result


class PublicMediaCpuConfig(StrictModel):
    maximum_media_per_run: int = Field(default=32, ge=1, le=128)
    maximum_media_bytes: int = Field(
        default=512 * 1_024 * 1_024,
        ge=1_024 * 1_024,
        le=512 * 1_024 * 1_024,
    )
    maximum_attempts_per_media: int = Field(default=3, ge=1, le=10)
    timeout_seconds: int = Field(default=120, ge=2, le=1_800)
    maximum_photo_pixels: int = Field(default=50_000_000, ge=1, le=100_000_000)
    maximum_photo_dimension: int = Field(default=2_048, ge=256, le=4_096)
    keyframes: VideoKeyframeConfig = Field(default_factory=VideoKeyframeConfig)


class PublicMediaAnalysisRunReceipt(StrictModel):
    analysis_id: SafeIdentifierV2
    source_revision_sha256: Sha256HexV2
    eligible_media_count: int = Field(ge=0)
    already_processed_count: int = Field(ge=0)
    attempted_count: int = Field(ge=0)
    succeeded_count: int = Field(ge=0)
    partial_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    remaining_count: int = Field(ge=0)
    raw_public_media_stored: Literal[False] = False
    raw_keyframes_stored: Literal[False] = False
    transcripts_stored: Literal[False] = False


class PublicMediaCpuWorker:
    def __init__(
        self,
        *,
        repository: EventEvidenceRepository,
        publisher: PublicMediaAnalysisPublisher,
        broker: ResearchBroker,
        broker_control_token: str,
        detector: TransientFrameDetector,
        transcription_provider: TranscriptionProvider | None,
        evidence_provider: MultimodalEvidenceProvider | None,
        audio_extractor: AudioTrackExtractor,
        config: PublicMediaCpuConfig | None = None,
        clock: Any = lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        self._publisher = publisher
        self._broker = broker
        self._broker_control_token = broker_control_token
        self._detector = detector
        self._transcription_provider = transcription_provider
        self._evidence_provider = evidence_provider
        self._audio_extractor = audio_extractor
        self._config = config or PublicMediaCpuConfig()
        self._clock = clock

    @staticmethod
    def _domain_policy(
        durable: DurableEventEvidence,
        source_url: str,
    ) -> tuple[str, dict[str, Any]]:
        policies = durable.research_source_policies or {}
        host = (urlsplit(source_url).hostname or "").casefold().rstrip(".")
        matches = [
            (domain, policy)
            for domain, policy in policies.items()
            if host == domain or host.endswith(f".{domain}")
        ]
        if not matches:
            raise ValueError("public media source escaped the durable domain policy")
        return max(matches, key=lambda item: len(item[0]))

    def _broker_session(self, durable: DurableEventEvidence) -> tuple[str, BrokerPolicy]:
        policies = durable.research_source_policies or {}
        templates = durable.research_search_templates or {}
        configured = self._broker.configure(
            {
                "control_token": self._broker_control_token,
                "policy": {
                    "allowed_domains": sorted(policies),
                    "search_templates": templates,
                    "max_fetch_bytes": 16 * 1_024 * 1_024,
                    "max_media_fetch_bytes": self._config.maximum_media_bytes,
                    "timeout_seconds": self._config.timeout_seconds,
                },
            }
        )
        token = str(configured["session_token"])
        return token, self._broker._session({"session_token": token})

    @staticmethod
    def _journal(
        *,
        stage: str,
        outcome: str,
        detail: str,
        identity: str,
        occurred_at: datetime,
        source_url: str,
        error_code: str | None = None,
        retryable: bool = False,
        provider_id: str | None = None,
        model_revision: str | None = None,
        prompt_revision: str | None = None,
    ) -> dict[str, Any]:
        return {
            "entry_id": _stable_id("JOURNAL-MEDIA", identity),
            "stage": stage,
            "outcome": outcome,
            "error_code": error_code,
            "detail": detail[:1_000],
            "source_url": source_url,
            "occurred_at": occurred_at.isoformat(),
            "retryable": retryable,
            "provider_id": provider_id,
            "model_revision": model_revision,
            "prompt_revision": prompt_revision,
        }

    def _detect_artifacts(
        self,
        *,
        media: BackendResearchMedia,
        artifacts: tuple[VideoKeyframeArtifact, ...],
        journal: list[dict[str, Any]],
        occurred_at: datetime,
    ) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        for artifact in artifacts:
            try:
                result = self._detector.detect(
                    media_id=artifact.media.media_id,
                    content_type="image/png",
                    content=artifact.data,
                )
                provider_run = result.provider_run
                observations.append(
                    {
                        "observation_id": _stable_id(
                            "OBS-MEDIA",
                            f"{artifact.media.media_id}:{provider_run.model_version}",
                        ),
                        "media_id": media.media_id,
                        "keyframe_id": artifact.media.media_id,
                        "frame_index": artifact.frame_index,
                        "timestamp_seconds": artifact.timestamp_seconds,
                        "frame_sha256": artifact.media.sha256,
                        "detector_provider_id": provider_run.provider_id,
                        "detector_model_id": provider_run.model_id,
                        "detector_model_revision": provider_run.model_version,
                        "detections": [
                            {
                                "detection_id": detection.detection_id,
                                "label": detection.detection_class,
                                "score": detection.score,
                                "x_min": detection.bbox[0],
                                "y_min": detection.bbox[1],
                                "x_max": detection.bbox[2],
                                "y_max": detection.bbox[3],
                            }
                            for detection in result.detections
                        ],
                        "abstained": False,
                        "reason_codes": [],
                        "frame_binary_stored": False,
                    }
                )
            except Exception as exc:
                observations.append(
                    {
                        "observation_id": _stable_id(
                            "OBS-MEDIA",
                            f"{artifact.media.media_id}:detector-failed",
                        ),
                        "media_id": media.media_id,
                        "keyframe_id": artifact.media.media_id,
                        "frame_index": artifact.frame_index,
                        "timestamp_seconds": artifact.timestamp_seconds,
                        "frame_sha256": artifact.media.sha256,
                        "detector_provider_id": "yolo-fire-smoke-cpu",
                        "detector_model_id": "unknown",
                        "detector_model_revision": "unavailable",
                        "detections": [],
                        "abstained": True,
                        "reason_codes": [type(exc).__name__[:128]],
                        "frame_binary_stored": False,
                    }
                )
        detected = sum(1 for item in observations if not item["abstained"])
        journal.append(
            self._journal(
                stage="visual_detection",
                outcome="success" if detected == len(observations) else "partial",
                detail=(
                    f"Analyzed {detected}/{len(observations)} transient keyframes; "
                    "all frame bytes were discarded."
                ),
                identity=f"visual:{media.media_id}:{detected}:{len(observations)}",
                occurred_at=occurred_at,
                source_url=media.source_url,
                retryable=detected != len(observations),
                provider_id="yolo-fire-smoke-cpu",
            )
        )
        return observations

    def _video_observations(
        self,
        *,
        durable: DurableEventEvidence,
        media: BackendResearchMedia,
        path: Path,
        journal: list[dict[str, Any]],
        occurred_at: datetime,
    ) -> tuple[list[dict[str, Any]], tuple[VideoKeyframeArtifact, ...]]:
        event = EventEvidenceV1(
            event_id=durable.event.event_id,
            time_window=durable.event.time_window,
            sources=tuple(
                source for source in durable.event.sources if source.source_id == media.source_id
            ),
            media=(media,),
            needs_human_review=True,
        )
        run = VideoKeyframeExtractor(
            decoder=OpenCvVideoFrameDecoder(path_resolver=lambda _media: path),
            config=self._config.keyframes,
        ).run(event)
        if not run.artifacts:
            raise RuntimeError("public video produced no usable keyframe")
        return (
            self._detect_artifacts(
                media=media,
                artifacts=run.artifacts,
                journal=journal,
                occurred_at=occurred_at,
            ),
            run.artifacts,
        )

    def _photo_observation(
        self,
        *,
        media: BackendResearchMedia,
        path: Path,
        journal: list[dict[str, Any]],
        occurred_at: datetime,
    ) -> tuple[list[dict[str, Any]], tuple[VideoKeyframeArtifact, ...]]:
        with Image.open(path) as opened:
            width, height = opened.size
            if width <= 0 or height <= 0 or width * height > self._config.maximum_photo_pixels:
                raise ValueError("public photo dimensions exceed the transient decode limit")
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail(
            (
                self._config.maximum_photo_dimension,
                self._config.maximum_photo_dimension,
            ),
            Image.Resampling.LANCZOS,
        )
        while True:
            stream = BytesIO()
            image.save(stream, format="PNG", compress_level=6)
            content = stream.getvalue()
            if len(content) <= _MAX_TRANSIENT_PNG_BYTES:
                break
            next_size = (max(256, image.width * 3 // 4), max(256, image.height * 3 // 4))
            if next_size == image.size:
                raise ValueError("public photo cannot fit the transient detector byte limit")
            image = image.resize(next_size, Image.Resampling.LANCZOS)
        digest = sha256(content).hexdigest()
        artifact = VideoKeyframeArtifact(
            media=EvidenceMedia(
                media_id=_stable_id("FRAME-PHOTO", f"{media.media_id}:{digest}"),
                source_id=media.source_id,
                media_group_id=media.media_group_id,
                origin_id=media.origin_id,
                kind="keyframe",
                sha256=digest,
                captured_at=media.captured_at,
                parent_media_id=media.media_id,
            ),
            frame_index=0,
            timestamp_seconds=0,
            format="png",
            data=content,
        )
        artifacts = (artifact,)
        return (
            self._detect_artifacts(
                media=media,
                artifacts=artifacts,
                journal=journal,
                occurred_at=occurred_at,
            ),
            artifacts,
        )

    def _transcribe(
        self,
        *,
        media: BackendResearchMedia,
        path: Path,
        temporary: Path,
        journal: list[dict[str, Any]],
        occurred_at: datetime,
    ) -> TransientTranscript | None:
        if media.kind == "photo":
            return None
        provider = self._transcription_provider
        if provider is None:
            journal.append(
                self._journal(
                    stage="transcription",
                    outcome="not_provided",
                    error_code="transcription_provider_disabled",
                    detail="No managed transcription provider is enabled for this run.",
                    identity=f"transcription:disabled:{media.media_id}",
                    occurred_at=occurred_at,
                    source_url=media.source_url,
                    retryable=True,
                )
            )
            return None
        input_path = path
        content_type = media.content_type
        if media.kind == "video":
            try:
                input_path = self._audio_extractor.extract(
                    path,
                    temporary / f"{media.media_id}.wav",
                )
                content_type = "audio/wav"
            except TranscriptionProviderError as exc:
                journal.append(
                    self._journal(
                        stage="transcription",
                        outcome="missing" if exc.code == "audio_track_not_available" else "failed",
                        error_code=exc.code,
                        detail="The public video exposed no usable transient audio track.",
                        identity=f"transcription:{exc.code}:{media.media_id}",
                        occurred_at=occurred_at,
                        source_url=media.source_url,
                        retryable=exc.retryable,
                        provider_id=provider.provider_id,
                        model_revision=provider.model_revision,
                    )
                )
                return None
        try:
            transcript = provider.transcribe(input_path, content_type=content_type)
        except TranscriptionProviderError as exc:
            journal.append(
                self._journal(
                    stage="transcription",
                    outcome="failed",
                    error_code=exc.code,
                    detail="The managed transcription provider returned no verified transcript.",
                    identity=f"transcription:{exc.code}:{media.media_id}",
                    occurred_at=occurred_at,
                    source_url=media.source_url,
                    retryable=exc.retryable,
                    provider_id=provider.provider_id,
                    model_revision=provider.model_revision,
                )
            )
            return None
        journal.append(
            self._journal(
                stage="transcription",
                outcome="partial" if transcript.partial else "success",
                error_code="transcription_partial" if transcript.partial else None,
                detail="The transient transcript was hashed, analyzed in memory and discarded.",
                identity=f"transcription:success:{media.media_id}:{transcript.transcript_sha256}",
                occurred_at=occurred_at,
                source_url=media.source_url,
                retryable=transcript.partial,
                provider_id=transcript.provider_id,
                model_revision=transcript.model_revision,
            )
        )
        return transcript

    def _extract_claims(
        self,
        *,
        durable: DurableEventEvidence,
        media: BackendResearchMedia,
        transcript: TransientTranscript | None,
        artifacts: tuple[VideoKeyframeArtifact, ...],
        observations: Sequence[Mapping[str, Any]],
        journal: list[dict[str, Any]],
        occurred_at: datetime,
    ) -> tuple[list[dict[str, Any]], str | None]:
        provider = self._evidence_provider
        if provider is None:
            journal.append(
                self._journal(
                    stage="text_analysis",
                    outcome="not_provided",
                    error_code="media_evidence_provider_disabled",
                    detail="No managed VL evidence provider is enabled for media-derived claims.",
                    identity=f"media-evidence:disabled:{media.media_id}",
                    occurred_at=occurred_at,
                    source_url=media.source_url,
                    retryable=True,
                )
            )
            return [], None
        source = next(item for item in durable.event.sources if item.source_id == media.source_id)
        _domain, policy = self._domain_policy(durable, str(source.source_url))
        allowed_claim_types = tuple(str(item) for item in policy["claim_types"])
        ranked = sorted(
            zip(artifacts, observations, strict=True),
            key=lambda pair: (
                -max(
                    (float(item["score"]) for item in pair[1]["detections"]),
                    default=0,
                ),
                pair[0].frame_index,
            ),
        )[:4]
        images = tuple(
            TransientEvidenceImage(
                media_id=artifact.media.media_id,
                content_type="image/png",
                sha256=artifact.media.sha256,
                content=artifact.data,
                public_content=True,
            )
            for artifact, _observation in ranked
        )
        detection_summary = json.dumps(
            [
                {
                    "keyframe_id": item["keyframe_id"],
                    "timestamp_seconds": item["timestamp_seconds"],
                    "detections": item["detections"],
                }
                for item in observations
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        content = (
            transcript.text[:95_000] + "\nVisual detections:\n" + detection_summary
            if transcript is not None
            else "Visual detections:\n" + detection_summary
        )[:100_000]
        try:
            extraction = provider.extract(
                MultimodalEvidenceDocument(
                    source_id=media.source_id,
                    source_url=AnyHttpUrl(media.source_url),
                    publisher=source.publisher,
                    published_at=source.published_at,
                    content_sha256=sha256(content.encode("utf-8")).hexdigest(),
                    content_type="text/plain",
                    content_role="transcript" if transcript is not None else "page",
                    transient_content=content,
                    images=images,
                    public_content=True,
                ),
                allowed_claim_types=allowed_claim_types,
            )
        except MultimodalEvidenceProviderError as exc:
            journal.append(
                self._journal(
                    stage="text_analysis",
                    outcome="failed",
                    error_code=exc.code,
                    detail="The managed VL provider returned no valid media-derived claim ticket.",
                    identity=f"media-evidence:{exc.code}:{media.media_id}",
                    occurred_at=occurred_at,
                    source_url=media.source_url,
                    retryable=exc.retryable,
                    provider_id=provider.provider_id,
                )
            )
            return [], None
        claims: list[dict[str, Any]] = []
        for claim in extraction.claims:
            identity = (
                f"{media.source_id}:{media.media_id}:{claim.claim_type}:"
                f"{claim.text}:{claim.observed_at}:{extraction.model_revision}"
            )
            claims.append(
                {
                    "claim_id": _stable_id("CLAIM-MEDIA", identity),
                    "source_id": media.source_id,
                    "claim_type": claim.claim_type,
                    "text": claim.text,
                    "observed_at": (
                        claim.observed_at.isoformat() if claim.observed_at is not None else None
                    ),
                    "confidence": claim.confidence,
                    "evidence_media_ids": [media.media_id],
                    **({"surface_area": claim.surface_area.model_dump(mode="json")}
                       if claim.surface_area is not None else {}),
                }
            )
        journal.append(
            self._journal(
                stage="text_analysis",
                outcome="partial" if extraction.partial else "success",
                error_code="media_evidence_partial" if extraction.partial else None,
                detail=f"Produced {len(claims)} sourced claim tickets from transient media.",
                identity=(
                    f"media-evidence:success:{media.media_id}:{extraction.model_revision}:"
                    f"{len(claims)}"
                ),
                occurred_at=occurred_at,
                source_url=media.source_url,
                retryable=extraction.partial,
                provider_id=extraction.provider_id,
                model_revision=extraction.model_revision,
                prompt_revision=extraction.prompt_revision,
            )
        )
        return claims, extraction.prompt_revision

    def _payload(
        self,
        *,
        durable: DurableEventEvidence,
        media: BackendResearchMedia,
        policy: BrokerPolicy,
        batch_id: str,
        attempt: int,
    ) -> dict[str, Any]:
        occurred_at = self._clock()
        journal: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="fireviewer-public-media-") as raw_dir:
            temporary = Path(raw_dir)
            suffix = Path(urlsplit(media.source_url).path).suffix[:10] or ".bin"
            path = temporary / f"source{suffix}"
            self._broker.materialize_transient_file(
                {"arguments": {"url": media.source_url, "store": False}},
                policy,
                destination=path,
                expected_sha256=media.sha256,
                expected_size_bytes=media.size_bytes,
                expected_content_type=media.content_type,
            )
            journal.append(
                self._journal(
                    stage="media_fetch",
                    outcome="success",
                    detail=(
                        "The public media ticket was streamed, verified and kept only temporarily."
                    ),
                    identity=f"media-fetch:{batch_id}:{attempt}",
                    occurred_at=occurred_at,
                    source_url=media.source_url,
                )
            )
            observations: list[dict[str, Any]] = []
            artifacts: tuple[VideoKeyframeArtifact, ...] = ()
            if media.kind == "video":
                observations, artifacts = self._video_observations(
                    durable=durable,
                    media=media,
                    path=path,
                    journal=journal,
                    occurred_at=occurred_at,
                )
            elif media.kind == "photo":
                observations, artifacts = self._photo_observation(
                    media=media,
                    path=path,
                    journal=journal,
                    occurred_at=occurred_at,
                )
            transcript = self._transcribe(
                media=media,
                path=path,
                temporary=temporary,
                journal=journal,
                occurred_at=occurred_at,
            )
            claims, _prompt_revision = self._extract_claims(
                durable=durable,
                media=media,
                transcript=transcript,
                artifacts=artifacts,
                observations=observations,
                journal=journal,
                occurred_at=occurred_at,
            )
        transcription_receipts = []
        if transcript is not None:
            transcription_receipts.append(
                {
                    "receipt_id": _stable_id(
                        "TRANS-MEDIA",
                        f"{media.media_id}:{transcript.provider_id}:"
                        f"{transcript.model_revision}:{transcript.transcript_sha256}",
                    ),
                    "media_id": media.media_id,
                    "provider_id": transcript.provider_id,
                    "model_revision": transcript.model_revision,
                    "transcript_sha256": transcript.transcript_sha256,
                    "duration_seconds": transcript.duration_seconds,
                    "language": transcript.language,
                    "claim_ids": [item["claim_id"] for item in claims],
                    "partial": transcript.partial,
                    "transcript_stored": False,
                    "audio_binary_stored": False,
                }
            )
        outcomes = {str(item["outcome"]) for item in journal}
        outcome: Literal["success", "partial"] = (
            "partial" if outcomes & {"failed", "partial", "not_provided"} else "success"
        )
        return {
            "schema_version": "research-media-analysis-1.0",
            "candidate_id": durable.event.event_id,
            "source_revision_sha256": durable.source_revision_sha256,
            "batch_id": batch_id,
            "media_id": media.media_id,
            "media_sha256": media.sha256,
            "processor_id": PROCESSOR_ID,
            "processor_revision": PROCESSOR_REVISION,
            "analyzed_at": occurred_at.isoformat(),
            "outcome": outcome,
            "claims": claims,
            "keyframe_observations": observations,
            "transcription_receipts": transcription_receipts,
            "journal_entries": journal,
            "source_binary_stored": False,
        }

    def _failed_payload(
        self,
        *,
        durable: DurableEventEvidence,
        media: BackendResearchMedia,
        batch_id: str,
        exc: Exception,
    ) -> dict[str, Any]:
        occurred_at = self._clock()
        return {
            "schema_version": "research-media-analysis-1.0",
            "candidate_id": durable.event.event_id,
            "source_revision_sha256": durable.source_revision_sha256,
            "batch_id": batch_id,
            "media_id": media.media_id,
            "media_sha256": media.sha256,
            "processor_id": PROCESSOR_ID,
            "processor_revision": PROCESSOR_REVISION,
            "analyzed_at": occurred_at.isoformat(),
            "outcome": "failed",
            "claims": [],
            "keyframe_observations": [],
            "transcription_receipts": [],
            "journal_entries": [
                self._journal(
                    stage="media_fetch",
                    outcome="failed",
                    error_code=type(exc).__name__[:128],
                    detail=f"Public-media processing failed with {type(exc).__name__}.",
                    identity=f"media-failed:{batch_id}:{type(exc).__name__}",
                    occurred_at=occurred_at,
                    source_url=media.source_url,
                    retryable=True,
                )
            ],
            "source_binary_stored": False,
        }

    def run_analysis(self, analysis_id: str) -> PublicMediaAnalysisRunReceipt:
        durable = self._repository.read(analysis_id)
        if durable.research_target_kind != "incident_day":
            raise ValueError("public-media worker requires an incident-day target")
        if durable.research_progress is None or not durable.research_progress.completed:
            raise RuntimeError("source collection must complete before public-media analysis")
        eligible = tuple(
            sorted(
                (
                    item
                    for item in durable.research_media_tickets
                    if item.kind in {"photo", "video", "audio"}
                    and item.size_bytes <= self._config.maximum_media_bytes
                ),
                key=lambda item: item.media_id,
            )
        )
        batches = durable.research_media_analysis_batches
        completed_media = {
            item.media_id
            for item in batches
            if item.media_sha256
            == next(
                (media.sha256 for media in eligible if media.media_id == item.media_id),
                None,
            )
            and item.processor_id == PROCESSOR_ID
            and item.processor_revision == PROCESSOR_REVISION
            and item.outcome == "success"
        }
        attempts = {
            media.media_id: sum(
                1
                for item in batches
                if item.media_id == media.media_id
                and item.processor_id == PROCESSOR_ID
                and item.processor_revision == PROCESSOR_REVISION
            )
            for media in eligible
        }
        pending = tuple(
            item
            for item in eligible
            if item.media_id not in completed_media
            and attempts[item.media_id] < self._config.maximum_attempts_per_media
        )
        selected = pending[: self._config.maximum_media_per_run]
        token, policy = self._broker_session(durable)
        succeeded = 0
        partial = 0
        failed = 0
        attempted_media_ids: set[str] = set()
        successful_media_ids: set[str] = set()
        revision = durable.source_revision_sha256
        try:
            for media in selected:
                attempted_media_ids.add(media.media_id)
                attempt = attempts[media.media_id] + 1
                batch_id = _stable_id(
                    "MEDIA-BATCH",
                    f"{analysis_id}:{media.media_id}:{media.sha256}:{PROCESSOR_REVISION}:{attempt}",
                )
                current = durable
                if current.source_revision_sha256 != revision:
                    current = replace(durable, source_revision_sha256=revision)
                try:
                    payload = self._payload(
                        durable=current,
                        media=media,
                        policy=policy,
                        batch_id=batch_id,
                        attempt=attempt,
                    )
                except Exception as exc:
                    payload = self._failed_payload(
                        durable=current,
                        media=media,
                        batch_id=batch_id,
                        exc=exc,
                    )
                receipt = self._publisher.publish(
                    candidate_id=analysis_id,
                    payload=payload,
                )
                revision = receipt.source_revision_sha256
                outcome = str(payload["outcome"])
                if outcome == "success":
                    succeeded += 1
                    successful_media_ids.add(media.media_id)
                elif outcome == "partial":
                    partial += 1
                else:
                    failed += 1
        finally:
            self._broker.revoke(
                {"control_token": self._broker_control_token, "session_token": token}
            )
        remaining = sum(
            1
            for media in eligible
            if media.media_id not in completed_media
            and media.media_id not in successful_media_ids
            and (attempts[media.media_id] + (1 if media.media_id in attempted_media_ids else 0))
            < self._config.maximum_attempts_per_media
        )
        return PublicMediaAnalysisRunReceipt(
            analysis_id=analysis_id,
            source_revision_sha256=revision,
            eligible_media_count=len(eligible),
            already_processed_count=len(completed_media),
            attempted_count=len(selected),
            succeeded_count=succeeded,
            partial_count=partial,
            failed_count=failed,
            remaining_count=remaining,
            raw_public_media_stored=False,
            raw_keyframes_stored=False,
            transcripts_stored=False,
        )


__all__ = [
    "PROCESSOR_ID",
    "PROCESSOR_REVISION",
    "HttpTransientYoloDetector",
    "PublicMediaAnalysisRunReceipt",
    "PublicMediaCpuConfig",
    "PublicMediaCpuWorker",
    "TransientFrameDetector",
]
