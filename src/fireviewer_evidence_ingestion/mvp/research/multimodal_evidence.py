"""Interchangeable multimodal extraction for sourced public EventEvidence."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from html.parser import HTMLParser
from io import BytesIO
from typing import Any, Literal, Protocol, cast
from urllib.parse import urlsplit

import boto3
import httpx
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import BotoCoreError, ClientError
from PIL import Image, UnidentifiedImageError
from pydantic import AnyHttpUrl, Field, model_validator

from fireviewer_contracts.contracts import SafeIdentifierV2, Sha256HexV2, StrictModel
from fireviewer_evidence_ingestion.mvp.research.surface_area import ExtractedSurfaceArea

_PROMPT = """
You extract explicitly supported wildfire facts from one public source content item and its images.
Return exactly one JSON object with this shape:
{"claims":[{"claim_type":"one allowed value","text":"short paraphrase",
"observed_at":"ISO-8601 timestamp with timezone or null","confidence":0.0,
"evidence_media_ids":["optional supplied media id"]}],"partial":false}

Rules:
- Use only the supplied source content and images. Never add outside knowledge.
- claim_type must be one of allowed_claim_types.
- text must be a concise paraphrase, never a copied passage.
- confidence measures extraction faithfulness, not whether the reported fact is true.
- Do not accept, reject, rank or generate a GPS candidate.
- An evidence_media_id is allowed only when that exact image supports the claim.
- Keep contradictory statements as separate claims.
- For an explicitly dated area claim, optionally add surface_area with component
  (affected/active), scope (incident/episode), accumulation (cumulative/incremental),
  qualifier (exact/approximate/minimum/maximum/interval), value_ha/lower_ha/upper_ha
  and timezone-aware valid_from/valid_until. Omit it if the period or meaning is unknown.
- Never invent bounds, tolerances, dates or an exact qualifier for an approximate area.
- Use an empty claims list when no supported fact is present.
- No Markdown and no fields beyond the schema.
""".strip()
PROMPT_REVISION = sha256(_PROMPT.encode("utf-8")).hexdigest()


class MultimodalEvidenceProviderError(RuntimeError):
    def __init__(self, code: str, *, retryable: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class TransientEvidenceImage(StrictModel):
    media_id: SafeIdentifierV2
    content_type: Literal["image/jpeg", "image/png", "image/webp", "image/avif"]
    sha256: Sha256HexV2
    content: bytes = Field(min_length=1, max_length=8 * 1_024 * 1_024, repr=False)
    public_content: Literal[True] = True

    @model_validator(mode="after")
    def validate_digest(self) -> TransientEvidenceImage:
        if sha256(self.content).hexdigest() != self.sha256:
            raise ValueError("transient image digest mismatch")
        return self


class MultimodalEvidenceDocument(StrictModel):
    source_id: SafeIdentifierV2
    source_url: AnyHttpUrl
    publisher: str = Field(min_length=1, max_length=500)
    published_at: datetime | None = None
    content_sha256: Sha256HexV2
    content_type: Literal["text/html", "text/plain", "application/json"]
    content_role: Literal["page", "transcript"] = "page"
    transient_content: str = Field(min_length=1, max_length=100_000, repr=False)
    images: tuple[TransientEvidenceImage, ...] = Field(default=(), max_length=4, repr=False)
    public_content: Literal[True] = True

    @model_validator(mode="after")
    def validate_images(self) -> MultimodalEvidenceDocument:
        media_ids = [item.media_id for item in self.images]
        if len(media_ids) != len(set(media_ids)):
            raise ValueError("transient evidence image identifiers must be unique")
        if sum(len(item.content) for item in self.images) > 12 * 1_024 * 1_024:
            raise ValueError("transient evidence images exceed the request byte budget")
        return self


class ExtractedMultimodalClaim(StrictModel):
    claim_type: SafeIdentifierV2
    text: str = Field(min_length=1, max_length=1_000)
    observed_at: datetime | None = None
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    evidence_media_ids: tuple[SafeIdentifierV2, ...] = Field(default=(), max_length=4)
    surface_area: ExtractedSurfaceArea | None = None


class MultimodalEvidenceExtraction(StrictModel):
    provider_id: SafeIdentifierV2
    model_revision: str = Field(min_length=1, max_length=255)
    prompt_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    claims: tuple[ExtractedMultimodalClaim, ...] = Field(default=(), max_length=64)
    partial: bool = False


class MultimodalEvidenceProvider(Protocol):
    provider_id: str

    def extract(
        self,
        document: MultimodalEvidenceDocument,
        *,
        allowed_claim_types: Sequence[str],
    ) -> MultimodalEvidenceExtraction: ...


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() in {"script", "style", "svg", "noscript", "template"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style", "svg", "noscript", "template"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        normalized = " ".join(data.split())
        if normalized:
            self._parts.append(normalized)

    def text(self) -> str:
        return "\n".join(self._parts)


def _transient_visible_text(document: MultimodalEvidenceDocument, maximum: int) -> str:
    if document.content_type != "text/html":
        return document.transient_content[:maximum]
    parser = _VisibleTextParser()
    try:
        parser.feed(document.transient_content)
        parser.close()
    except Exception as exc:
        raise MultimodalEvidenceProviderError("multimodal_html_parse_failed") from exc
    visible = parser.text().strip()
    if not visible:
        raise MultimodalEvidenceProviderError("multimodal_document_empty")
    return visible[:maximum]


def _json_object(value: str) -> Mapping[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(value):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise MultimodalEvidenceProviderError("multimodal_invalid_json")


class _ProviderClaimPayload(StrictModel):
    claim_type: SafeIdentifierV2
    text: str = Field(min_length=1, max_length=1_000)
    observed_at: datetime | None = None
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    evidence_media_ids: tuple[SafeIdentifierV2, ...] = Field(default=(), max_length=4)
    surface_area: ExtractedSurfaceArea | None = None

    @model_validator(mode="after")
    def validate_observed_at(self) -> _ProviderClaimPayload:
        if self.observed_at is not None and (
            self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None
        ):
            raise ValueError("multimodal claim timestamp must include a timezone")
        return self


class _ProviderPayload(StrictModel):
    claims: tuple[_ProviderClaimPayload, ...] = Field(default=(), max_length=64)
    partial: bool = False


class BedrockPixtralConfig(StrictModel):
    region_name: str = Field(default="eu-west-3", pattern=r"^[a-z]{2}-[a-z]+-\d$")
    model_id: str = Field(
        default="eu.mistral.pixtral-large-2502-v1:0",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,255}$",
    )
    model_revision: str = Field(default="mistral.pixtral-large-2502-v1:0", min_length=3)
    maximum_input_characters: int = Field(default=40_000, ge=1_000, le=100_000)
    maximum_output_tokens: int = Field(default=2_048, ge=256, le=4_096)
    maximum_images: int = Field(default=4, ge=1, le=4)


class BedrockConverseClient(Protocol):
    def converse(self, **kwargs: Any) -> Mapping[str, Any]: ...


class InvocationLimitedBedrockClient:
    """Process-local fail-closed reservation for an explicitly authorized paid call lot."""

    def __init__(self, client: BedrockConverseClient, *, maximum_invocations: int) -> None:
        if maximum_invocations < 1:
            raise ValueError("Bedrock invocation budget must be positive")
        self._client = client
        self._maximum_invocations = maximum_invocations
        self._reserved_invocations = 0
        self._lock = threading.Lock()

    @property
    def remaining_invocations(self) -> int:
        with self._lock:
            return self._maximum_invocations - self._reserved_invocations

    def converse(self, **kwargs: Any) -> Mapping[str, Any]:
        with self._lock:
            if self._reserved_invocations >= self._maximum_invocations:
                raise MultimodalEvidenceProviderError(
                    "bedrock_paid_invocation_budget_exhausted",
                    retryable=False,
                )
            self._reserved_invocations += 1
        return self._client.converse(**kwargs)


class AzureManagedIdentityWebTokenProvider:
    """Read an Azure Container Apps managed-identity token from its loopback broker."""

    def __init__(self, *, audience: str, managed_identity_client_id: str) -> None:
        self._audience = audience
        self._managed_identity_client_id = managed_identity_client_id

    def __call__(self) -> str:
        endpoint = os.environ.get("IDENTITY_ENDPOINT", "").strip()
        identity_header = os.environ.get("IDENTITY_HEADER", "").strip()
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or not identity_header
        ):
            raise MultimodalEvidenceProviderError(
                "azure_managed_identity_endpoint_unavailable", retryable=True
            )
        try:
            response = httpx.get(
                endpoint,
                params={
                    "resource": self._audience,
                    "api-version": "2019-08-01",
                    "client_id": self._managed_identity_client_id,
                },
                headers={"X-IDENTITY-HEADER": identity_header},
                timeout=5.0,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise MultimodalEvidenceProviderError(
                "azure_managed_identity_token_failed", retryable=True
            ) from exc
        token = payload.get("access_token") if isinstance(payload, Mapping) else None
        if not isinstance(token, str) or len(token) < 100:
            raise MultimodalEvidenceProviderError(
                "azure_managed_identity_token_invalid", retryable=True
            )
        return token


class AzureFederatedBedrockClient:
    """Cache short-lived AWS credentials obtained from a dedicated Azure identity."""

    def __init__(
        self,
        *,
        role_arn: str,
        region_name: str,
        web_token_provider: Callable[[], str],
        role_session_name: str = "fireviewer-source-acquisition",
        sts_client: Any | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._role_arn = role_arn
        self._region_name = region_name
        self._web_token_provider = web_token_provider
        self._role_session_name = role_session_name
        self._sts = sts_client or boto3.client(
            "sts",
            region_name=region_name,
            config=BotocoreConfig(
                retries={"mode": "adaptive", "total_max_attempts": 4},
                connect_timeout=5,
                read_timeout=10,
            ),
        )
        self._clock = clock
        self._client: BedrockConverseClient | None = None
        self._expires_at: datetime | None = None
        self._lock = threading.Lock()

    def _fresh_client(self) -> BedrockConverseClient:
        with self._lock:
            now = self._clock()
            if (
                self._client is not None
                and self._expires_at is not None
                and self._expires_at - now > timedelta(minutes=5)
            ):
                return self._client
            try:
                assumed = self._sts.assume_role_with_web_identity(
                    RoleArn=self._role_arn,
                    RoleSessionName=self._role_session_name,
                    WebIdentityToken=self._web_token_provider(),
                    DurationSeconds=3600,
                )
                credentials = assumed["Credentials"]
                expires_at = credentials["Expiration"]
                if not isinstance(expires_at, datetime):
                    raise TypeError("AWS STS expiration is invalid")
                session = boto3.Session(
                    aws_access_key_id=str(credentials["AccessKeyId"]),
                    aws_secret_access_key=str(credentials["SecretAccessKey"]),
                    aws_session_token=str(credentials["SessionToken"]),
                    region_name=self._region_name,
                )
                client = session.client(
                    "bedrock-runtime",
                    config=BotocoreConfig(
                        retries={"mode": "adaptive", "total_max_attempts": 4},
                        connect_timeout=5,
                        read_timeout=90,
                    ),
                )
            except (BotoCoreError, ClientError, KeyError, TypeError) as exc:
                raise MultimodalEvidenceProviderError(
                    "aws_bedrock_federation_failed", retryable=True
                ) from exc
            self._client = cast(BedrockConverseClient, client)
            self._expires_at = expires_at.astimezone(UTC)
            return self._client

    def converse(self, **kwargs: Any) -> Mapping[str, Any]:
        return self._fresh_client().converse(**kwargs)


def _bedrock_image(image: TransientEvidenceImage) -> dict[str, Any]:
    format_by_type = {
        "image/jpeg": "jpeg",
        "image/png": "png",
        "image/webp": "webp",
    }
    image_format = format_by_type.get(image.content_type)
    content = image.content
    if image_format is None:
        try:
            with Image.open(BytesIO(content)) as source:
                output = BytesIO()
                source.convert("RGB").save(output, format="PNG", optimize=True)
                content = output.getvalue()
        except (UnidentifiedImageError, OSError) as exc:
            raise MultimodalEvidenceProviderError("bedrock_image_conversion_failed") from exc
        image_format = "png"
    return {"image": {"format": image_format, "source": {"bytes": content}}}


def _validated_extraction(
    *,
    provider_id: str,
    model_revision: str,
    parsed: _ProviderPayload,
    allowed: tuple[str, ...],
    supplied_media_ids: set[str],
    partial: bool,
) -> MultimodalEvidenceExtraction:
    allowed_set = set(allowed)
    if any(claim.claim_type not in allowed_set for claim in parsed.claims):
        raise MultimodalEvidenceProviderError("multimodal_claim_type_rejected")
    if any(
        not set(claim.evidence_media_ids).issubset(supplied_media_ids) for claim in parsed.claims
    ):
        raise MultimodalEvidenceProviderError("multimodal_media_reference_rejected")
    unique: dict[tuple[str, str, str, tuple[str, ...]], ExtractedMultimodalClaim] = {}
    for claim in parsed.claims:
        normalized_text = " ".join(claim.text.split())
        media_ids = tuple(dict.fromkeys(claim.evidence_media_ids))
        key = (
            claim.claim_type,
            normalized_text.casefold(),
            claim.observed_at.isoformat() if claim.observed_at is not None else "",
            media_ids,
        )
        unique.setdefault(
            key,
            ExtractedMultimodalClaim(
                claim_type=claim.claim_type,
                text=normalized_text,
                observed_at=claim.observed_at,
                confidence=claim.confidence,
                evidence_media_ids=media_ids,
                surface_area=claim.surface_area,
            ),
        )
    return MultimodalEvidenceExtraction(
        provider_id=provider_id,
        model_revision=model_revision,
        prompt_revision=PROMPT_REVISION,
        claims=tuple(unique.values()),
        partial=partial or parsed.partial,
    )


class BedrockPixtralMultimodalProvider:
    """Pixtral Large through the provider-neutral Bedrock Converse API."""

    provider_id = "aws-bedrock-pixtral"

    def __init__(
        self,
        config: BedrockPixtralConfig,
        *,
        client: BedrockConverseClient,
    ) -> None:
        self.config = config
        self._client = client

    def extract(
        self,
        document: MultimodalEvidenceDocument,
        *,
        allowed_claim_types: Sequence[str],
    ) -> MultimodalEvidenceExtraction:
        allowed = tuple(dict.fromkeys(allowed_claim_types))
        if not allowed or len(allowed) > 32:
            raise MultimodalEvidenceProviderError("multimodal_claim_policy_invalid")
        visible_text = _transient_visible_text(document, self.config.maximum_input_characters)
        selected_images = document.images[: self.config.maximum_images]
        content: list[dict[str, Any]] = [
            {
                "text": json.dumps(
                    {
                        "source_id": document.source_id,
                        "source_url": str(document.source_url),
                        "publisher": document.publisher,
                        "published_at": (
                            document.published_at.isoformat()
                            if document.published_at is not None
                            else None
                        ),
                        "allowed_claim_types": allowed,
                        "content_role": document.content_role,
                        "public_content_text": visible_text,
                        "supplied_media_ids": [item.media_id for item in selected_images],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            }
        ]
        content.extend(_bedrock_image(image) for image in selected_images)
        try:
            response = self._client.converse(
                modelId=self.config.model_id,
                system=[{"text": _PROMPT}],
                messages=[{"role": "user", "content": content}],
                inferenceConfig={
                    "maxTokens": self.config.maximum_output_tokens,
                    "temperature": 0,
                },
            )
        except MultimodalEvidenceProviderError:
            raise
        except (BotoCoreError, ClientError) as exc:
            code = (
                str(exc.response.get("Error", {}).get("Code", ""))
                if isinstance(exc, ClientError)
                else ""
            )
            retryable = code in {
                "InternalServerException",
                "ModelNotReadyException",
                "ServiceUnavailableException",
                "ThrottlingException",
            }
            raise MultimodalEvidenceProviderError(
                "bedrock_pixtral_request_failed", retryable=retryable
            ) from exc
        try:
            message = response["output"]["message"]
            blocks = message["content"]
            text = "\n".join(
                str(block["text"])
                for block in blocks
                if isinstance(block, Mapping) and isinstance(block.get("text"), str)
            )
            parsed = _ProviderPayload.model_validate(_json_object(text))
            stop_reason = str(response.get("stopReason", ""))
        except MultimodalEvidenceProviderError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise MultimodalEvidenceProviderError("bedrock_pixtral_invalid_response") from exc
        return _validated_extraction(
            provider_id=self.provider_id,
            model_revision=self.config.model_revision,
            parsed=parsed,
            allowed=allowed,
            supplied_media_ids={item.media_id for item in selected_images},
            partial=stop_reason == "max_tokens",
        )


__all__ = [
    "PROMPT_REVISION",
    "AzureFederatedBedrockClient",
    "AzureManagedIdentityWebTokenProvider",
    "BedrockConverseClient",
    "BedrockPixtralConfig",
    "BedrockPixtralMultimodalProvider",
    "ExtractedMultimodalClaim",
    "InvocationLimitedBedrockClient",
    "MultimodalEvidenceDocument",
    "MultimodalEvidenceExtraction",
    "MultimodalEvidenceProvider",
    "MultimodalEvidenceProviderError",
    "TransientEvidenceImage",
]
