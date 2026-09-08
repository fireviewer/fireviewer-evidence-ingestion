"""Deterministic CPU public-source acquisition with durable page checkpoints."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Literal, Protocol, cast
from urllib.parse import urlsplit

from pydantic import AnyHttpUrl, Field, model_validator

from fireviewer_contracts.contracts import SafeIdentifierV2, StrictModel
from fireviewer_evidence_ingestion.mvp.research.multimodal_evidence import (
    MultimodalEvidenceDocument,
    MultimodalEvidenceProvider,
    MultimodalEvidenceProviderError,
    TransientEvidenceImage,
)
from fireviewer_contracts.mvp.supervision.backend_event_evidence import (
    BackendResearchEvidencePublisher,
    BackendResearchEvidenceReceipt,
    EventEvidenceRepository,
)
from fireviewer_evidence_ingestion.research_broker import BrokerPolicy, ResearchBroker

_DEFAULT_TEXT_CLAIM_TYPES = (
    "incident_status",
    "ignition",
    "location_report",
    "observation_time",
    "fire_progression",
    "fire_resumption",
    "fire_fixed",
    "fire_contained",
    "fire_controlled",
    "fire_extinguished",
    "area_burned",
    "response_resources",
    "evacuation",
    "public_instruction",
    "road_closure",
    "damage_report",
    "casualty_report",
    "cause_report",
    "weather_condition",
)


class SourceDomainPolicy(StrictModel):
    publisher: str = Field(min_length=1, max_length=500)
    source_type: Literal[
        "official",
        "press",
        "social",
        "witness",
        "satellite",
        "panoramax",
        "metadata",
        "other",
    ]
    independence_weight: float = Field(ge=0, le=1, allow_inf_nan=False)
    claim_types: tuple[SafeIdentifierV2, ...] = Field(
        default=_DEFAULT_TEXT_CLAIM_TYPES,
        min_length=1,
        max_length=32,
    )


class SourceAcquisitionPlan(StrictModel):
    candidate_id: SafeIdentifierV2
    plan_id: SafeIdentifierV2
    wave_number: int = Field(default=1, ge=1, le=16)
    wave_focus: tuple[SafeIdentifierV2, ...] = Field(min_length=1, max_length=32)
    queries: tuple[str, ...] = Field(min_length=1, max_length=100)
    allowed_domains: tuple[str, ...] = Field(min_length=1, max_length=200)
    source_policies: dict[str, SourceDomainPolicy]
    search_provider_domain: str = Field(min_length=1, max_length=255)
    search_template: str = Field(min_length=16, max_length=2_048)
    media_ticket_limit: int = Field(default=100, ge=1, le=2_048)
    video_ticket_limit: int = Field(default=30, ge=0, le=512)
    max_source_pages: int = Field(default=200, ge=1, le=10_000)
    convergence_zero_yield_waves: int = Field(default=2, ge=2, le=5)
    results_per_page: int = Field(default=20, ge=1, le=50)
    media_per_source: int = Field(default=8, ge=1, le=20)
    max_multimodal_analyses_per_run: int = Field(default=20, ge=1, le=100)
    max_pages_per_run: int = Field(default=5, ge=1, le=50)
    max_fetch_bytes: int = Field(default=16 * 1_024 * 1_024, ge=65_536, le=64 * 1_024 * 1_024)
    max_media_fetch_bytes: int = Field(
        default=512 * 1_024 * 1_024,
        ge=65_536,
        le=512 * 1_024 * 1_024,
    )
    timeout_seconds: int = Field(default=20, ge=2, le=120)

    @model_validator(mode="after")
    def validate_plan(self) -> SourceAcquisitionPlan:
        normalized_domains = tuple(item.casefold().rstrip(".") for item in self.allowed_domains)
        if len(normalized_domains) != len(set(normalized_domains)):
            raise ValueError("source acquisition domains must be unique")
        if set(self.source_policies) != set(normalized_domains):
            raise ValueError("every source acquisition domain requires a policy")
        if self.max_media_fetch_bytes < self.max_fetch_bytes:
            raise ValueError("media fetch limit cannot be smaller than the page fetch limit")
        if self.video_ticket_limit > self.media_ticket_limit:
            raise ValueError("video ticket limit cannot exceed the total media ticket limit")
        provider = self.search_provider_domain.casefold().rstrip(".")
        if provider in normalized_domains:
            raise ValueError("the search provider must be separate from source domains")
        template_host = (urlsplit(self.search_template).hostname or "").casefold().rstrip(".")
        if template_host != provider or self.search_template.count("{query}") != 1:
            raise ValueError("the search template does not match its provider")
        return self

    @property
    def revision(self) -> str:
        payload = self.model_dump(mode="json", exclude={"candidate_id", "max_pages_per_run"})
        return _canonical_sha256(payload)


class SourceAcquisitionRunReceipt(StrictModel):
    candidate_id: SafeIdentifierV2
    plan_id: SafeIdentifierV2
    plan_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    wave_number: int = Field(ge=1, le=16)
    wave_focus: tuple[SafeIdentifierV2, ...] = Field(min_length=1, max_length=32)
    pages_published: int = Field(ge=0)
    source_count: int = Field(ge=0)
    claim_count: int = Field(ge=0)
    media_count: int = Field(ge=0)
    completed: bool
    media_ticket_limit: int = Field(ge=1, le=2_048)
    safety_limit_reached: bool
    converged: bool
    zero_yield_wave_streak: int = Field(ge=0, le=100)
    coverage_ready: bool
    next_cursor: str | None = None
    source_revision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    errors: tuple[str, ...] = Field(default=(), max_length=200)


class ResearchEvidencePublisher(Protocol):
    def publish(
        self,
        *,
        candidate_id: str,
        payload: Mapping[str, Any],
    ) -> BackendResearchEvidenceReceipt: ...


def _canonical_sha256(payload: object) -> str:
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}-{sha256(value.encode('utf-8')).hexdigest()[:24]}"


def _encode_cursor(query_index: int, provider_cursor: str | None) -> str:
    raw = json.dumps(
        {"query_index": query_index, "provider_cursor": provider_cursor},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(value: str | None) -> tuple[int, str | None]:
    if value is None:
        return 0, None
    try:
        padded = value + ("=" * (-len(value) % 4))
        payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        query_index = int(payload["query_index"])
        provider_cursor = payload.get("provider_cursor")
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("durable research cursor is invalid") from exc
    if query_index < 0 or (provider_cursor is not None and not isinstance(provider_cursor, str)):
        raise ValueError("durable research cursor is invalid")
    return query_index, provider_cursor


def _aware_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _policy_for(host: str, plan: SourceAcquisitionPlan) -> SourceDomainPolicy:
    normalized = host.casefold().rstrip(".")
    matches = [
        (domain, policy)
        for domain, policy in plan.source_policies.items()
        if normalized == domain or normalized.endswith(f".{domain}")
    ]
    if not matches:
        raise ValueError("source domain escaped the acquisition plan")
    return max(matches, key=lambda item: len(item[0]))[1]


class CpuSourceAcquisitionWorker:
    """Collect public pages without a model and checkpoint every search page."""

    def __init__(
        self,
        *,
        repository: EventEvidenceRepository,
        publisher: ResearchEvidencePublisher,
        broker: ResearchBroker,
        broker_control_token: str,
        multimodal_evidence_provider: MultimodalEvidenceProvider | None = None,
        clock: Any = lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        self._publisher = publisher
        self._broker = broker
        self._broker_control_token = broker_control_token
        self._multimodal_evidence_provider = multimodal_evidence_provider
        self._clock = clock

    def _extract_multimodal_claims(
        self,
        *,
        fetched: Mapping[str, Any],
        source_id: str,
        source_url: str,
        publisher: str,
        published_at: datetime | None,
        source_policy: SourceDomainPolicy,
        images: tuple[TransientEvidenceImage, ...],
        retrieved_at: datetime,
        plan: SourceAcquisitionPlan,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        provider = self._multimodal_evidence_provider
        if provider is None:
            return [], []
        transient_content = fetched.get("text")
        if not isinstance(transient_content, str) or not transient_content.strip():
            return [], [
                {
                    "entry_id": _stable_id("JOURNAL-TEXT", f"missing:{source_url}"),
                    "stage": "text_analysis",
                    "outcome": "not_provided",
                    "error_code": "source_text_not_provided",
                    "detail": "The fetched public page exposed no transient content to analyze.",
                    "source_url": source_url,
                    "occurred_at": retrieved_at.isoformat(),
                    "retryable": False,
                    "provider_id": None,
                    "model_revision": None,
                    "prompt_revision": plan.revision,
                }
            ]
        try:
            extraction = provider.extract(
                MultimodalEvidenceDocument(
                    source_id=source_id,
                    source_url=AnyHttpUrl(source_url),
                    publisher=publisher,
                    published_at=published_at,
                    content_sha256=str(fetched["sha256"]),
                    content_type="text/html",
                    transient_content=transient_content,
                    images=images,
                    public_content=True,
                ),
                allowed_claim_types=source_policy.claim_types,
            )
        except MultimodalEvidenceProviderError as exc:
            return [], [
                {
                    "entry_id": _stable_id(
                        "JOURNAL-TEXT",
                        f"failed:{source_url}:{exc.code}",
                    ),
                    "stage": "text_analysis",
                    "outcome": "failed",
                    "error_code": exc.code,
                    "detail": "The multimodal evidence provider returned no valid claim ticket.",
                    "source_url": source_url,
                    "occurred_at": retrieved_at.isoformat(),
                    "retryable": exc.retryable,
                    "provider_id": getattr(provider, "provider_id", None),
                    "model_revision": None,
                    "prompt_revision": plan.revision,
                }
            ]
        claims = []
        for claim in extraction.claims:
            observed = claim.observed_at.isoformat() if claim.observed_at is not None else ""
            media_refs = ":".join(claim.evidence_media_ids)
            claim_key = f"{source_id}:{claim.claim_type}:{claim.text}:{observed}:{media_refs}"
            claims.append(
                {
                    "claim_id": _stable_id("CLAIM-TEXT", claim_key),
                    "source_id": source_id,
                    "claim_type": claim.claim_type,
                    "text": claim.text,
                    "observed_at": claim.observed_at.isoformat()
                    if claim.observed_at is not None
                    else None,
                    "confidence": claim.confidence,
                    "evidence_media_ids": list(claim.evidence_media_ids),
                    **({"surface_area": claim.surface_area.model_dump(mode="json")}
                       if claim.surface_area is not None else {}),
                }
            )
        outcome: Literal["success", "partial"] = "partial" if extraction.partial else "success"
        journal_entry = {
            "entry_id": _stable_id(
                "JOURNAL-TEXT",
                f"{outcome}:{source_url}:{extraction.model_revision}:"
                f"{extraction.prompt_revision}:{len(claims)}",
            ),
            "stage": "text_analysis",
            "outcome": outcome,
            "error_code": "text_analysis_partial" if extraction.partial else None,
            "detail": (
                f"Extracted {len(claims)} structured claim tickets with "
                f"{len(images)} transient public image inputs."
            ),
            "source_url": source_url,
            "occurred_at": retrieved_at.isoformat(),
            "retryable": extraction.partial,
            "provider_id": extraction.provider_id,
            "model_revision": extraction.model_revision,
            "prompt_revision": extraction.prompt_revision,
        }
        return claims, [journal_entry]

    @staticmethod
    def _broker_policy(plan: SourceAcquisitionPlan) -> dict[str, object]:
        # This CPU path records hashes and public provenance only. Omitting every
        # upload field disables binary persistence inside the shared broker.
        return {
            "allowed_domains": list(plan.allowed_domains),
            "search_templates": {
                plan.search_provider_domain: plan.search_template,
            },
            "max_fetch_bytes": plan.max_fetch_bytes,
            "max_media_fetch_bytes": plan.max_media_fetch_bytes,
            "timeout_seconds": plan.timeout_seconds,
        }

    def _collect_search_page(
        self,
        *,
        plan: SourceAcquisitionPlan,
        policy: BrokerPolicy,
        query: str,
        provider_cursor: str | None,
        known_source_urls: set[str],
        known_source_content_hashes: set[str],
        known_media_hashes: set[str],
        research_media_count: int,
        research_video_count: int,
        remaining_multimodal_analyses: int,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        str | None,
        list[dict[str, Any]],
        int,
    ]:
        searched = self._broker.search(
            {
                "arguments": {
                    "domain": plan.search_provider_domain,
                    "query": query,
                    "cursor": provider_cursor,
                    "limit": plan.results_per_page,
                }
            },
            policy,
        )
        sources: list[dict[str, Any]] = []
        claims: list[dict[str, Any]] = []
        media: list[dict[str, Any]] = []
        journal: list[dict[str, Any]] = []
        retrieved_at = self._clock()
        multimodal_analyses = 0
        if not searched["links"]:
            journal.append(
                {
                    "entry_id": _stable_id(
                        "JOURNAL-WEB",
                        f"search:missing:{query}:{provider_cursor or ''}",
                    ),
                    "stage": "search",
                    "outcome": "missing",
                    "error_code": "search_results_missing",
                    "detail": "The search provider returned no allowlisted source URL.",
                    "source_url": None,
                    "occurred_at": retrieved_at.isoformat(),
                    "retryable": False,
                    "provider_id": plan.search_provider_domain,
                    "model_revision": None,
                    "prompt_revision": plan.revision,
                }
            )
        for link in searched["links"]:
            page_url = str(link["url"])
            if page_url in known_source_urls:
                continue
            try:
                fetched = self._broker.fetch(
                    {"arguments": {"url": page_url, "store": False}},
                    policy,
                )
            except Exception as exc:
                journal.append(
                    {
                        "entry_id": _stable_id(
                            "JOURNAL-WEB",
                            f"page_fetch:{page_url}:{type(exc).__name__}:{exc}",
                        ),
                        "stage": "page_fetch",
                        "outcome": "failed",
                        "error_code": type(exc).__name__[:128],
                        "detail": f"The source page could not be fetched: {exc}"[:1_000],
                        "source_url": page_url,
                        "occurred_at": retrieved_at.isoformat(),
                        "retryable": True,
                        "provider_id": None,
                        "model_revision": None,
                        "prompt_revision": plan.revision,
                    }
                )
                continue
            content_type = str(fetched.get("content_type", ""))
            if content_type != "text/html":
                journal.append(
                    {
                        "entry_id": _stable_id(
                            "JOURNAL-WEB",
                            f"page_parse:rejected:{page_url}:{content_type}",
                        ),
                        "stage": "page_parse",
                        "outcome": "rejected",
                        "error_code": "unsupported_source_content_type",
                        "detail": "The source result was not an HTML article page.",
                        "source_url": page_url,
                        "occurred_at": retrieved_at.isoformat(),
                        "retryable": False,
                        "provider_id": None,
                        "model_revision": None,
                        "prompt_revision": plan.revision,
                    }
                )
                continue
            content_sha256 = str(fetched.get("sha256", ""))
            if len(content_sha256) != 64:
                journal.append(
                    {
                        "entry_id": _stable_id(
                            "JOURNAL-WEB",
                            f"page_parse:hash_invalid:{page_url}:{content_sha256}",
                        ),
                        "stage": "page_parse",
                        "outcome": "rejected",
                        "error_code": "source_content_hash_invalid",
                        "detail": "The source page did not provide a valid immutable content hash.",
                        "source_url": page_url,
                        "occurred_at": retrieved_at.isoformat(),
                        "retryable": False,
                        "provider_id": None,
                        "model_revision": None,
                        "prompt_revision": plan.revision,
                    }
                )
                continue
            if content_sha256 in known_source_content_hashes:
                journal.append(
                    {
                        "entry_id": _stable_id(
                            "JOURNAL-WEB",
                            f"page_parse:syndicated:{page_url}:{content_sha256}",
                        ),
                        "stage": "page_parse",
                        "outcome": "rejected",
                        "error_code": "syndicated_or_republished_content",
                        "detail": "An identical article body already belongs to this evidence set.",
                        "source_url": page_url,
                        "occurred_at": retrieved_at.isoformat(),
                        "retryable": False,
                        "provider_id": None,
                        "model_revision": None,
                        "prompt_revision": plan.revision,
                    }
                )
                continue
            final_url = str(fetched.get("url") or page_url)
            host = (urlsplit(final_url).hostname or "").casefold().rstrip(".")
            source_policy = _policy_for(host, plan)
            metadata = fetched.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            source_id = _stable_id("SRC-WEB", final_url)
            published_at = _aware_datetime(
                metadata.get("article:published_time")
                or metadata.get("date")
                or metadata.get("datepublished")
            )
            publisher = str(metadata.get("og:site_name") or source_policy.publisher)[:500]
            sources.append(
                {
                    "source_id": source_id,
                    "origin_id": _stable_id("ORIGIN-WEB", final_url),
                    "source_url": final_url,
                    "publisher": publisher,
                    "published_at": published_at.isoformat() if published_at else None,
                    "retrieved_at": retrieved_at.isoformat(),
                    "source_type": source_policy.source_type,
                    "independence_weight": source_policy.independence_weight,
                    "content_sha256": content_sha256,
                }
            )
            known_source_urls.add(final_url)
            known_source_content_hashes.add(content_sha256)
            source_media = 0
            transient_images: list[TransientEvidenceImage] = []
            raw_media_links = fetched.get("media_links")
            candidates = raw_media_links if isinstance(raw_media_links, list) else []
            for media_url_value in candidates:
                if (
                    research_media_count + len(media) >= plan.media_ticket_limit
                    or source_media >= plan.media_per_source
                ):
                    break
                media_url = str(media_url_value)
                try:
                    media_fetch, transient_bytes = self._broker.fetch_transient_media(
                        {"arguments": {"url": media_url, "store": False}},
                        policy,
                        maximum_in_memory_bytes=8 * 1_024 * 1_024,
                    )
                except Exception as exc:
                    journal.append(
                        {
                            "entry_id": _stable_id(
                                "JOURNAL-WEB",
                                f"media_fetch:{media_url}:{type(exc).__name__}:{exc}",
                            ),
                            "stage": "media_fetch",
                            "outcome": "failed",
                            "error_code": type(exc).__name__[:128],
                            "detail": f"The referenced public media could not be hashed: {exc}"[
                                :1_000
                            ],
                            "source_url": media_url,
                            "occurred_at": retrieved_at.isoformat(),
                            "retryable": True,
                            "provider_id": None,
                            "model_revision": None,
                            "prompt_revision": plan.revision,
                        }
                    )
                    continue
                media_type = str(media_fetch.get("content_type", ""))
                kind: Literal["photo", "video", "audio"]
                if media_type in {
                    "image/avif",
                    "image/jpeg",
                    "image/png",
                    "image/webp",
                }:
                    kind = "photo"
                elif media_type in {"video/mp4", "video/quicktime", "video/webm"}:
                    kind = "video"
                elif media_type in {
                    "audio/aac",
                    "audio/flac",
                    "audio/m4a",
                    "audio/mpeg",
                    "audio/ogg",
                    "audio/wav",
                    "audio/webm",
                }:
                    kind = "audio"
                else:
                    continue
                if kind == "video" and (
                    research_video_count + sum(item["kind"] == "video" for item in media)
                    >= plan.video_ticket_limit
                ):
                    continue
                digest = str(media_fetch.get("sha256", ""))
                if len(digest) != 64 or digest in known_media_hashes:
                    continue
                known_media_hashes.add(digest)
                source_media += 1
                media_id = _stable_id("MEDIA-WEB", digest)
                media.append(
                    {
                        "media_id": media_id,
                        "source_id": source_id,
                        "media_group_id": _stable_id("GROUP-WEB", final_url),
                        "origin_id": _stable_id("ORIGIN-MEDIA-WEB", media_url),
                        "kind": kind,
                        "sha256": digest,
                        "captured_at": published_at.isoformat() if published_at else None,
                        "source_url": str(media_fetch.get("url") or media_url),
                        "content_type": media_type,
                        "size_bytes": int(media_fetch.get("size_bytes", 0)),
                    }
                )
                if kind == "photo" and transient_bytes is not None and len(transient_images) < 4:
                    transient_images.append(
                        TransientEvidenceImage(
                            media_id=media_id,
                            content_type=cast(
                                Literal[
                                    "image/jpeg",
                                    "image/png",
                                    "image/webp",
                                    "image/avif",
                                ],
                                media_type,
                            ),
                            sha256=digest,
                            content=transient_bytes,
                            public_content=True,
                        )
                    )
            if multimodal_analyses < remaining_multimodal_analyses:
                extracted_claims, multimodal_journal = self._extract_multimodal_claims(
                    fetched=fetched,
                    source_id=source_id,
                    source_url=final_url,
                    publisher=publisher,
                    published_at=published_at,
                    source_policy=source_policy,
                    images=tuple(transient_images),
                    retrieved_at=retrieved_at,
                    plan=plan,
                )
                if self._multimodal_evidence_provider is not None:
                    multimodal_analyses += 1
            else:
                extracted_claims = []
                multimodal_journal = [
                    {
                        "entry_id": _stable_id(
                            "JOURNAL-TEXT",
                            f"budget:{plan.revision}:{source_id}",
                        ),
                        "stage": "text_analysis",
                        "outcome": "not_provided",
                        "error_code": "multimodal_run_call_budget_exhausted",
                        "detail": "The bounded multimodal call budget was exhausted for this run.",
                        "source_url": final_url,
                        "occurred_at": retrieved_at.isoformat(),
                        "retryable": True,
                        "provider_id": getattr(
                            self._multimodal_evidence_provider, "provider_id", None
                        ),
                        "model_revision": None,
                        "prompt_revision": plan.revision,
                    }
                ]
            claims.extend(extracted_claims)
            journal.extend(multimodal_journal)
            # Only tickets and provider receipts leave this method. Article text and
            # public image bytes are discarded after this iteration.
        return (
            sources,
            claims,
            media,
            searched.get("next_cursor"),
            journal,
            multimodal_analyses,
        )

    def run(self, plan: SourceAcquisitionPlan) -> SourceAcquisitionRunReceipt:
        durable = self._repository.read(plan.candidate_id)
        progress = durable.research_progress
        if progress is not None and (
            progress.plan_id != plan.plan_id or progress.plan_revision != plan.revision
        ):
            if not (
                durable.research_target_kind == "incident_day"
                and progress.completed
                and plan.wave_number == progress.wave_number + 1
                and (
                    not progress.converged
                    or (
                        durable.incident_day_coverage is not None
                        and not durable.incident_day_coverage.documentary_ready
                    )
                )
            ):
                raise ValueError("EventEvidence already contains another research plan")
            progress = None
        if progress is not None and progress.completed:
            return SourceAcquisitionRunReceipt(
                candidate_id=plan.candidate_id,
                plan_id=plan.plan_id,
                plan_revision=plan.revision,
                wave_number=plan.wave_number,
                wave_focus=plan.wave_focus,
                pages_published=0,
                source_count=len(durable.event.sources),
                claim_count=len(durable.event.claims),
                media_count=len(durable.event.media),
                completed=True,
                media_ticket_limit=plan.media_ticket_limit,
                safety_limit_reached=progress.safety_limit_reached,
                converged=progress.converged,
                zero_yield_wave_streak=progress.zero_yield_wave_streak,
                coverage_ready=progress.coverage_ready,
                source_revision_sha256=durable.source_revision_sha256,
            )

        query_index, provider_cursor = _decode_cursor(
            progress.next_cursor if progress is not None else None
        )
        page_number = (progress.page_count if progress is not None else 0) + 1
        revision = durable.source_revision_sha256
        known_source_urls = {
            str(item.source_url) for item in durable.event.sources if item.source_url is not None
        }
        known_source_content_hashes = {
            item.content_sha256
            for item in durable.research_sources
            if item.content_sha256 is not None
        }
        known_media_hashes = {item.sha256 for item in durable.event.media}
        public_source_ids = {
            item.source_id for item in durable.event.sources if item.source_url is not None
        }
        research_media_count = sum(
            1 for item in durable.event.media if item.source_id in public_source_ids
        )
        research_video_count = sum(
            1
            for item in durable.event.media
            if item.source_id in public_source_ids and item.kind == "video"
        )
        configured = self._broker.configure(
            {
                "control_token": self._broker_control_token,
                "policy": self._broker_policy(plan),
            }
        )
        session_token = str(configured["session_token"])
        policy = self._broker._session({"session_token": session_token})
        pages_published = 0
        errors: list[str] = []
        completed = False
        next_cursor: str | None = None
        last_receipt: BackendResearchEvidenceReceipt | None = None
        multimodal_analyses = 0
        zero_yield_wave_streak = progress.zero_yield_wave_streak if progress is not None else 0
        safety_limit_reached = False
        converged = False
        try:
            while pages_published < plan.max_pages_per_run:
                if query_index >= len(plan.queries):
                    completed = True
                    sources: list[dict[str, Any]] = []
                    claims: list[dict[str, Any]] = []
                    media: list[dict[str, Any]] = []
                    journal_entries: list[dict[str, Any]] = []
                    next_cursor = None
                else:
                    try:
                        (
                            sources,
                            claims,
                            media,
                            provider_next,
                            journal_entries,
                            page_multimodal_analyses,
                        ) = self._collect_search_page(
                            plan=plan,
                            policy=policy,
                            query=plan.queries[query_index],
                            provider_cursor=provider_cursor,
                            known_source_urls=known_source_urls,
                            known_source_content_hashes=known_source_content_hashes,
                            known_media_hashes=known_media_hashes,
                            research_media_count=research_media_count,
                            research_video_count=research_video_count,
                            remaining_multimodal_analyses=(
                                plan.max_multimodal_analyses_per_run - multimodal_analyses
                            ),
                        )
                        multimodal_analyses += page_multimodal_analyses
                        research_media_count += len(media)
                        research_video_count += sum(item["kind"] == "video" for item in media)
                        if sources or claims or media:
                            zero_yield_wave_streak = 0
                        else:
                            zero_yield_wave_streak += 1
                    except Exception as exc:
                        occurred_at = self._clock()
                        sources, claims, media = [], [], []
                        provider_next = provider_cursor
                        journal_entries = [
                            {
                                "entry_id": _stable_id(
                                    "JOURNAL-WEB",
                                    f"search:failed:{page_number}:{query_index}:"
                                    f"{provider_cursor or ''}:{type(exc).__name__}:{exc}",
                                ),
                                "stage": "search",
                                "outcome": "failed",
                                "error_code": type(exc).__name__[:128],
                                "detail": f"The search page request failed: {exc}"[:1_000],
                                "source_url": None,
                                "occurred_at": occurred_at.isoformat(),
                                "retryable": True,
                                "provider_id": plan.search_provider_domain,
                                "model_revision": None,
                                "prompt_revision": plan.revision,
                            }
                        ]
                    errors.extend(
                        item["detail"]
                        for item in journal_entries
                        if item["outcome"] in {"failed", "partial", "missing", "not_provided"}
                    )
                    if (
                        research_media_count >= plan.media_ticket_limit
                        or page_number >= plan.max_source_pages
                    ):
                        completed = True
                        safety_limit_reached = True
                        next_cursor = None
                    elif provider_next is not None:
                        next_cursor = _encode_cursor(query_index, str(provider_next))
                    elif query_index + 1 < len(plan.queries):
                        next_cursor = _encode_cursor(query_index + 1, None)
                    else:
                        completed = True
                        next_cursor = None
                if completed:
                    converged = (
                        not safety_limit_reached
                        and zero_yield_wave_streak >= plan.convergence_zero_yield_waves
                    )
                if completed and not converged:
                    journal_entries.append(
                        {
                            "entry_id": _stable_id(
                                "JOURNAL-WEB",
                                f"coverage:partial:{plan.revision}:{page_number}:"
                                f"{zero_yield_wave_streak}:{safety_limit_reached}",
                            ),
                            "stage": "planning",
                            "outcome": "partial",
                            "error_code": (
                                "media_ticket_safety_limit_reached"
                                if safety_limit_reached
                                else "collection_not_converged"
                            ),
                            "detail": (
                                "The adaptive query wave ended before two consecutive "
                                "zero-yield searches confirmed collection convergence."
                            ),
                            "source_url": None,
                            "occurred_at": self._clock().isoformat(),
                            "retryable": False,
                            "provider_id": plan.search_provider_domain,
                            "model_revision": None,
                            "prompt_revision": plan.revision,
                        }
                    )
                page_id = _stable_id(
                    "PAGE-WEB",
                    f"{plan.revision}:{page_number}:{query_index}:{provider_cursor or ''}",
                )
                payload = {
                    "schema_version": "research-evidence-page-1.0",
                    "candidate_id": plan.candidate_id,
                    "source_revision_sha256": revision,
                    "plan_id": plan.plan_id,
                    "plan_revision": plan.revision,
                    "wave_number": plan.wave_number,
                    "wave_focus": list(plan.wave_focus),
                    "page_id": page_id,
                    "page_number": page_number,
                    "cursor": (
                        _encode_cursor(query_index, provider_cursor)
                        if query_index or provider_cursor is not None
                        else None
                    ),
                    "next_cursor": next_cursor,
                    "completed": completed,
                    "media_ticket_limit": plan.media_ticket_limit,
                    "safety_limit_reached": safety_limit_reached,
                    "converged": converged,
                    "zero_yield_wave_streak": zero_yield_wave_streak,
                    "coverage_ready": completed and converged,
                    "sources": sources,
                    "claims": claims,
                    "media": media,
                    "journal_entries": journal_entries,
                }
                last_receipt = self._publisher.publish(
                    candidate_id=plan.candidate_id,
                    payload=payload,
                )
                revision = last_receipt.source_revision_sha256
                pages_published += 1
                if completed:
                    break
                query_index, provider_cursor = _decode_cursor(next_cursor)
                page_number += 1
        finally:
            self._broker.revoke(
                {
                    "control_token": self._broker_control_token,
                    "session_token": session_token,
                }
            )
        if last_receipt is None:
            raise RuntimeError("source acquisition produced no durable page receipt")
        return SourceAcquisitionRunReceipt(
            candidate_id=plan.candidate_id,
            plan_id=plan.plan_id,
            plan_revision=plan.revision,
            wave_number=plan.wave_number,
            wave_focus=plan.wave_focus,
            pages_published=pages_published,
            source_count=last_receipt.source_count,
            claim_count=last_receipt.claim_count,
            media_count=last_receipt.media_count,
            completed=last_receipt.completed,
            media_ticket_limit=last_receipt.media_ticket_limit,
            safety_limit_reached=last_receipt.safety_limit_reached,
            converged=last_receipt.converged,
            zero_yield_wave_streak=last_receipt.zero_yield_wave_streak,
            coverage_ready=last_receipt.coverage_ready,
            next_cursor=last_receipt.next_cursor,
            source_revision_sha256=last_receipt.source_revision_sha256,
            errors=tuple(errors[:200]),
        )


def build_source_acquisition_worker(
    *,
    repository: EventEvidenceRepository,
    publisher: BackendResearchEvidencePublisher,
    broker: ResearchBroker,
    broker_control_token: str,
    multimodal_evidence_provider: MultimodalEvidenceProvider,
) -> CpuSourceAcquisitionWorker:
    return CpuSourceAcquisitionWorker(
        repository=repository,
        publisher=publisher,
        broker=broker,
        broker_control_token=broker_control_token,
        multimodal_evidence_provider=multimodal_evidence_provider,
    )


__all__ = [
    "CpuSourceAcquisitionWorker",
    "ResearchEvidencePublisher",
    "SourceAcquisitionPlan",
    "SourceAcquisitionRunReceipt",
    "SourceDomainPolicy",
    "build_source_acquisition_worker",
]
