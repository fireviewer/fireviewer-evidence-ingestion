"""Transient managed transcription; only hashes and derived claims may be persisted."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from typing import Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import Field, SecretStr, model_validator

from fireviewer_contracts.contracts import Sha256HexV2, StrictModel

_MAX_TRANSCRIPTION_RESPONSE_BYTES = 8 * 1_024 * 1_024


class TranscriptionProviderError(RuntimeError):
    def __init__(self, code: str, *, retryable: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class TransientTranscript(StrictModel):
    provider_id: str = Field(min_length=1, max_length=128)
    model_revision: str = Field(min_length=1, max_length=255)
    text: str = Field(min_length=1, max_length=500_000, repr=False)
    transcript_sha256: Sha256HexV2
    duration_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    language: str | None = Field(default=None, min_length=2, max_length=16)
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    partial: bool = False
    transcript_stored: bool = False
    audio_binary_stored: bool = False

    @model_validator(mode="after")
    def validate_transient_contract(self) -> TransientTranscript:
        if sha256(self.text.encode("utf-8")).hexdigest() != self.transcript_sha256:
            raise ValueError("transient transcript digest mismatch")
        if self.transcript_stored or self.audio_binary_stored:
            raise ValueError("transient transcription cannot retain source content")
        return self


class TranscriptionProvider(Protocol):
    provider_id: str
    model_revision: str

    def transcribe(self, path: Path, *, content_type: str) -> TransientTranscript: ...


class AudioTrackExtractor(Protocol):
    def extract(self, source: Path, destination: Path) -> Path: ...


class FfmpegAudioTrackExtractor:
    def __init__(self, *, executable: str = "ffmpeg", timeout_seconds: int = 1_800) -> None:
        self._executable = executable
        self._timeout_seconds = timeout_seconds

    def extract(self, source: Path, destination: Path) -> Path:
        if not source.is_file() or destination.exists() or not destination.parent.is_dir():
            raise TranscriptionProviderError("audio_extraction_path_invalid")
        try:
            result = subprocess.run(  # noqa: S603
                [
                    self._executable,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(destination),
                ],
                check=False,
                capture_output=True,
                timeout=self._timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            destination.unlink(missing_ok=True)
            raise TranscriptionProviderError(
                "audio_extraction_failed",
                retryable=isinstance(exc, subprocess.TimeoutExpired),
            ) from exc
        if result.returncode != 0 or not destination.is_file() or destination.stat().st_size <= 44:
            destination.unlink(missing_ok=True)
            raise TranscriptionProviderError("audio_track_not_available")
        return destination


class AzureSpeechFastTranscriptionProvider:
    """Azure Speech fast transcription with managed-identity bearer authentication."""

    provider_id = "azure-speech-fast-transcription"
    model_revision = "azure-speech-default-2025-10-15"

    def __init__(
        self,
        *,
        endpoint: str,
        token_provider: Callable[[], str] | None = None,
        subscription_key: SecretStr | None = None,
        locales: tuple[str, ...] = ("fr-FR",),
        timeout_seconds: float = 300,
        client: httpx.Client | None = None,
    ) -> None:
        parsed = urlsplit(endpoint.rstrip("/"))
        host = (parsed.hostname or "").casefold()
        if (
            parsed.scheme != "https"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in {None, 443}
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or not (
                host.endswith(".cognitiveservices.azure.com")
                or host.endswith(".api.cognitive.microsoft.com")
            )
        ):
            raise ValueError("Azure Speech endpoint must be an HTTPS resource origin")
        if (token_provider is None) == (subscription_key is None):
            raise ValueError("configure exactly one Azure Speech authentication method")
        if not locales or len(locales) > 10:
            raise ValueError("Azure Speech locales are invalid")
        self._endpoint = endpoint.rstrip("/")
        self._token_provider = token_provider
        self._subscription_key = subscription_key
        self._locales = locales
        self._timeout_seconds = timeout_seconds
        self._client = client

    def transcribe(self, path: Path, *, content_type: str) -> TransientTranscript:
        if not path.is_file() or not 0 < path.stat().st_size < 250 * 1_024 * 1_024:
            raise TranscriptionProviderError("azure_speech_input_invalid")
        headers = {"Accept": "application/json"}
        if self._token_provider is not None:
            headers["Authorization"] = f"Bearer {self._token_provider()}"
        elif self._subscription_key is not None:
            headers["Ocp-Apim-Subscription-Key"] = self._subscription_key.get_secret_value()
        owned_client = self._client is None
        client = self._client or httpx.Client(
            timeout=httpx.Timeout(self._timeout_seconds, connect=10),
            follow_redirects=False,
            trust_env=False,
        )
        try:
            with (
                path.open("rb") as stream,
                client.stream(
                    "POST",
                    self._endpoint
                    + "/speechtotext/transcriptions:transcribe?api-version=2025-10-15",
                    headers=headers,
                    files={"audio": (path.name, stream, content_type)},
                    data={
                        "definition": json.dumps(
                            {"locales": list(self._locales)},
                            separators=(",", ":"),
                        )
                    },
                ) as response,
            ):
                if response.is_redirect:
                    raise TranscriptionProviderError("azure_speech_redirect_forbidden")
                response.raise_for_status()
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > _MAX_TRANSCRIPTION_RESPONSE_BYTES:
                        raise TranscriptionProviderError("azure_speech_response_too_large")
            value = json.loads(body)
        except (httpx.HTTPError, OSError, ValueError) as exc:
            raise TranscriptionProviderError(
                "azure_speech_request_failed",
                retryable=isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)),
            ) from exc
        finally:
            if owned_client:
                client.close()
        if not isinstance(value, Mapping):
            raise TranscriptionProviderError("azure_speech_response_invalid")
        combined = value.get("combinedPhrases")
        phrases = value.get("phrases")
        texts = (
            [
                str(item["text"]).strip()
                for item in combined
                if isinstance(item, Mapping) and str(item.get("text", "")).strip()
            ]
            if isinstance(combined, list)
            else []
        )
        phrase_items = (
            [item for item in phrases if isinstance(item, Mapping)]
            if isinstance(phrases, list)
            else []
        )
        if not texts:
            texts = [
                str(item["text"]).strip()
                for item in phrase_items
                if str(item.get("text", "")).strip()
            ]
        text = "\n".join(texts).strip()
        if not text:
            raise TranscriptionProviderError("azure_speech_transcript_empty")
        language_values = [
            str(item["locale"]) for item in phrase_items if isinstance(item.get("locale"), str)
        ]
        scores = [
            float(item["confidence"])
            for item in phrase_items
            if isinstance(item.get("confidence"), (int, float))
            and 0 <= float(item["confidence"]) <= 1
        ]
        duration_ms = value.get("durationMilliseconds")
        duration_seconds = (
            float(duration_ms) / 1_000
            if isinstance(duration_ms, (int, float)) and duration_ms > 0
            else None
        )
        return TransientTranscript(
            provider_id=self.provider_id,
            model_revision=self.model_revision,
            text=text,
            transcript_sha256=sha256(text.encode("utf-8")).hexdigest(),
            duration_seconds=duration_seconds,
            language=(Counter(language_values).most_common(1)[0][0] if language_values else None),
            confidence=fmean(scores) if scores else None,
            partial=False,
            transcript_stored=False,
            audio_binary_stored=False,
        )


__all__ = [
    "AudioTrackExtractor",
    "AzureSpeechFastTranscriptionProvider",
    "FfmpegAudioTrackExtractor",
    "TranscriptionProvider",
    "TranscriptionProviderError",
    "TransientTranscript",
]
