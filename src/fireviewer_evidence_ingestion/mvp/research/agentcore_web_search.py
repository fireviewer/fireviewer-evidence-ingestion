from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Literal, Protocol
from urllib.parse import SplitResult, urlsplit, urlunsplit

from pydantic import AnyHttpUrl, Field

from fireviewer_contracts.contracts import StrictModel
from fireviewer_contracts.mvp.contracts import (
    Claim,
    EventEvidenceV1,
    EvidenceSource,
    Uncertainty,
)
from fireviewer_contracts.mvp.providers import (
    ProviderDescriptor,
    ProviderHealth,
    ResearchRequest,
)


class McpToolClient(Protocol):
    """Minimal MCP client boundary used by the AgentCore Gateway runtime."""

    def call_tool(self, *, name: str, arguments: dict[str, object]) -> dict[str, Any]: ...


class AgentCoreWebSearchConfig(StrictModel):
    region_name: Literal["us-east-1"] = "us-east-1"
    connector_version: Literal["1.2.0"] = "1.2.0"
    tool_name: str = Field(default="WebSearch", min_length=1, max_length=128)
    max_results: int = Field(default=20, ge=1, le=25)
    source_type: Literal["other"] = "other"
    independence_weight: float = Field(default=0.5, ge=0, le=1)
    snippet_confidence: float = Field(default=0.5, ge=0, le=1)


class _SearchResult(StrictModel):
    text: str = Field(min_length=1, max_length=9_000)
    url: str = Field(min_length=1, max_length=4_000)
    title: str | None = Field(default=None, min_length=1, max_length=900)
    publishedDate: str | None = Field(default=None, min_length=1, max_length=64)


def _identifier(prefix: str, value: str) -> str:
    return f"{prefix}-{sha256(value.encode('utf-8')).hexdigest()[:24]}"


def _canonical_http_url(value: str) -> tuple[str, str] | None:
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname.lower() if parsed.hostname is not None else ""
    if scheme not in {"http", "https"} or not hostname:
        return None

    try:
        port = parsed.port
    except ValueError:
        return None
    if port is None or (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        netloc = hostname
    else:
        netloc = f"{hostname}:{port}"
    path = parsed.path or "/"
    canonical = urlunsplit(SplitResult(scheme, netloc, path, parsed.query, ""))
    return canonical, hostname


def _parse_published_at(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _utc_iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _result_payloads(response: dict[str, Any]) -> tuple[list[object], int]:
    if response.get("isError") is True:
        return [], 0
    content = response.get("content")
    if not isinstance(content, list):
        return [], 1

    results: list[object] = []
    dropped = 0
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            dropped += 1
            continue
        text = block.get("text")
        if not isinstance(text, str):
            dropped += 1
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            dropped += 1
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            dropped += 1
            continue
        results.extend(payload["results"])
    return results, dropped


class AgentCoreWebSearchProvider:
    """Normalize AgentCore Web Search citations into additive EventEvidence."""

    def __init__(
        self,
        *,
        client: McpToolClient,
        config: AgentCoreWebSearchConfig | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.client = client
        self.config = config or AgentCoreWebSearchConfig()
        self.clock = clock or (lambda: datetime.now(UTC))
        self.descriptor = ProviderDescriptor(
            provider_id="aws-agentcore-web-search",
            provider_version=self.config.connector_version,
            config=self.config.model_dump(mode="json"),
            capabilities=("managed-web-search", "source-citations", "published-date-filter"),
        )

    def healthcheck(self) -> ProviderHealth:
        if not callable(getattr(self.client, "call_tool", None)):
            return ProviderHealth(
                status="unavailable",
                checked_at=self.clock(),
                reason_codes=("agentcore_mcp_client_unavailable",),
            )
        return ProviderHealth(status="healthy", checked_at=self.clock())

    def search(self, request: ResearchRequest) -> EventEvidenceV1:
        if len(request.query) > 200:
            raise ValueError("AgentCore Web Search query must contain at most 200 characters")
        existing = request.existing_evidence
        if existing is not None:
            if existing.event_id != request.event_id:
                raise ValueError("existing evidence belongs to a different event")
            if existing.time_window != request.time_window:
                raise ValueError("existing evidence disagrees with the research time window")

        arguments: dict[str, object] = {
            "query": request.query,
            "maxResults": self.config.max_results,
        }
        published_filter: dict[str, str] = {}
        if request.time_window.from_at is not None:
            published_filter["from"] = _utc_iso(request.time_window.from_at)
        if request.time_window.to_at is not None:
            published_filter["to"] = _utc_iso(request.time_window.to_at)
        if published_filter:
            arguments["filters"] = {"publishedDateFilter": published_filter}

        response = self.client.call_tool(name=self.config.tool_name, arguments=arguments)
        retrieved_at = self.clock()
        raw_results, dropped = _result_payloads(response)
        provider_error = response.get("isError") is True

        existing_urls = {
            canonical
            for source in (() if existing is None else existing.sources)
            if source.source_url is not None
            for normalized in (_canonical_http_url(str(source.source_url)),)
            if normalized is not None
            for canonical in (normalized[0],)
        }
        sources: list[EvidenceSource] = []
        claims: list[Claim] = []
        seen_urls = set(existing_urls)
        duplicate_count = 0

        for raw_result in raw_results:
            try:
                result = _SearchResult.model_validate(raw_result)
            except ValueError:
                dropped += 1
                continue
            normalized = _canonical_http_url(result.url)
            if normalized is None:
                dropped += 1
                continue
            canonical_url, hostname = normalized
            if canonical_url in seen_urls:
                duplicate_count += 1
                continue
            seen_urls.add(canonical_url)

            source_id = _identifier("SRC-WEB", canonical_url)
            origin_id = _identifier("ORG-WEB", hostname)
            published_at = _parse_published_at(result.publishedDate)
            title = result.title.strip() if result.title is not None else None
            claim_text = f"{title}\n{result.text}" if title else result.text
            sources.append(
                EvidenceSource(
                    source_id=source_id,
                    origin_id=origin_id,
                    source_url=AnyHttpUrl(canonical_url),
                    publisher=hostname,
                    published_at=published_at,
                    retrieved_at=retrieved_at,
                    source_type=self.config.source_type,
                    independence_weight=self.config.independence_weight,
                )
            )
            claims.append(
                Claim(
                    claim_id=_identifier("CLAIM-WEB", f"{source_id}:{claim_text}"),
                    source_id=source_id,
                    claim_type="web_search_result",
                    text=claim_text,
                    observed_at=published_at,
                    confidence=self.config.snippet_confidence,
                )
            )

        uncertainties = self._uncertainties(
            event_id=request.event_id,
            provider_error=provider_error,
            sources_found=len(sources),
            dropped=dropped,
            duplicate_count=duplicate_count,
        )
        return EventEvidenceV1(
            event_id=request.event_id,
            time_window=request.time_window,
            candidate_area=None if existing is None else existing.candidate_area,
            sources=tuple(sources),
            claims=tuple(claims),
            uncertainties=uncertainties,
            needs_human_review=True,
        )

    @staticmethod
    def _uncertainties(
        *,
        event_id: str,
        provider_error: bool,
        sources_found: int,
        dropped: int,
        duplicate_count: int,
    ) -> tuple[Uncertainty, ...]:
        messages: list[tuple[str, str]] = []
        if provider_error:
            messages.append(
                ("web_search_provider_error", "AgentCore Web Search returned an MCP error.")
            )
        elif sources_found == 0 and duplicate_count:
            messages.append(
                (
                    "web_search_no_new_results",
                    "All usable search results were already present in EventEvidence.",
                )
            )
        elif sources_found == 0:
            messages.append(
                ("web_search_no_usable_results", "Web Search returned no citable HTTP result.")
            )
        else:
            messages.append(
                (
                    "web_search_snippets_unverified",
                    "Search snippets are cited discovery evidence, not verified page contents.",
                )
            )
        if dropped:
            messages.append(
                (
                    "web_search_results_dropped",
                    f"{dropped} malformed or uncitable Web Search result blocks were dropped.",
                )
            )
        return tuple(
            Uncertainty(
                uncertainty_id=_identifier("UNC", f"{event_id}:{code}"),
                code=code,
                scope_type="event",
                scope_id=event_id,
                description=description,
            )
            for code, description in messages
        )


__all__ = [
    "AgentCoreWebSearchConfig",
    "AgentCoreWebSearchProvider",
    "McpToolClient",
]
