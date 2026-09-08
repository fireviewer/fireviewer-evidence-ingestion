from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from fireviewer_evidence_ingestion.mvp.research.transcription import (
    AzureSpeechFastTranscriptionProvider,
    TranscriptionProviderError,
)


def test_azure_fast_transcription_returns_only_transient_text_receipt(tmp_path: Path) -> None:
    audio = tmp_path / "briefing.wav"
    audio.write_bytes(b"RIFF" + (b"0" * 128))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer managed-identity-token"
        assert request.url.path == "/speechtotext/transcriptions:transcribe"
        assert request.url.params["api-version"] == "2025-10-15"
        assert b'"locales":["fr-FR"]' in request.content
        return httpx.Response(
            200,
            json={
                "durationMilliseconds": 2500,
                "combinedPhrases": [{"text": "Le feu est fixe."}],
                "phrases": [
                    {
                        "text": "Le feu est fixe.",
                        "locale": "fr-FR",
                        "confidence": 0.93,
                    }
                ],
            },
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        transcript = AzureSpeechFastTranscriptionProvider(
            endpoint="https://fireviewer-speech.cognitiveservices.azure.com",
            token_provider=lambda: "managed-identity-token",
            client=client,
        ).transcribe(audio, content_type="audio/wav")

    assert transcript.text == "Le feu est fixe."
    assert transcript.duration_seconds == 2.5
    assert transcript.language == "fr-FR"
    assert transcript.confidence == 0.93
    assert transcript.transcript_stored is False
    assert transcript.audio_binary_stored is False


def test_azure_fast_transcription_rejects_an_oversized_response(tmp_path: Path) -> None:
    audio = tmp_path / "briefing.wav"
    audio.write_bytes(b"RIFF" + (b"0" * 128))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"{" + (b" " * (8 * 1_024 * 1_024)),
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = AzureSpeechFastTranscriptionProvider(
            endpoint="https://fireviewer-speech.cognitiveservices.azure.com",
            token_provider=lambda: "managed-identity-token",
            client=client,
        )
        with pytest.raises(
            TranscriptionProviderError,
            match="azure_speech_response_too_large",
        ):
            provider.transcribe(audio, content_type="audio/wav")
