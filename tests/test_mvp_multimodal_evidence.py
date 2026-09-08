from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

import pytest

from fireviewer_evidence_ingestion.mvp.research.multimodal_evidence import (
    BedrockPixtralConfig,
    BedrockPixtralMultimodalProvider,
    MultimodalEvidenceDocument,
    MultimodalEvidenceProviderError,
    TransientEvidenceImage,
)


def _document() -> MultimodalEvidenceDocument:
    image = b"\xff\xd8\xffpublic-fire-image"
    return MultimodalEvidenceDocument(
        source_id="SRC-WEB-1",
        source_url="https://source.example/fire",
        publisher="Prefecture",
        published_at=datetime(2026, 8, 23, 12, tzinfo=UTC),
        content_sha256="a" * 64,
        content_type="text/html",
        transient_content=(
            "<html><script>RAW_SCRIPT_SECRET</script><body>"
            "Le feu a parcouru 120 hectares.</body></html>"
        ),
        images=(
            TransientEvidenceImage(
                media_id="MEDIA-WEB-1",
                content_type="image/jpeg",
                sha256=sha256(image).hexdigest(),
                content=image,
            ),
        ),
    )


class _BedrockClient:
    def __init__(self, claim_media_id: str = "MEDIA-WEB-1") -> None:
        self.claim_media_id = claim_media_id
        self.requests: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.requests.append(kwargs)
        return {
            "stopReason": "end_turn",
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "text": json.dumps(
                                {
                                    "claims": [
                                        {
                                            "claim_type": "area_burned",
                                            "text": (
                                                "La source rapporte 120 hectares parcourus."
                                            ),
                                            "observed_at": "2026-08-23T12:00:00Z",
                                            "confidence": 0.93,
                                            "surface_area": {
                                                "component": "affected",
                                                "qualifier": "approximate",
                                                "value_ha": 120,
                                                "valid_from": "2026-08-23T12:00:00Z",
                                                "valid_until": "2026-08-23T12:00:00Z",
                                            },
                                            "evidence_media_ids": [self.claim_media_id],
                                        }
                                    ],
                                    "partial": False,
                                }
                            )
                        }
                    ],
                }
            },
        }


def test_bedrock_pixtral_extracts_strict_multimodal_claims_with_converse() -> None:
    client = _BedrockClient()
    provider = BedrockPixtralMultimodalProvider(BedrockPixtralConfig(), client=client)

    result = provider.extract(_document(), allowed_claim_types=("area_burned",))

    request = client.requests[0]
    assert request["modelId"] == "eu.mistral.pixtral-large-2502-v1:0"
    assert request["inferenceConfig"] == {"maxTokens": 2048, "temperature": 0}
    content = request["messages"][0]["content"]
    text_payload = json.loads(content[0]["text"])
    assert text_payload["content_role"] == "page"
    assert "120 hectares" in text_payload["public_content_text"]
    assert "RAW_SCRIPT_SECRET" not in text_payload["public_content_text"]
    assert text_payload["supplied_media_ids"] == ["MEDIA-WEB-1"]
    assert content[1]["image"]["format"] == "jpeg"
    assert content[1]["image"]["source"]["bytes"].startswith(b"\xff\xd8\xff")
    assert result.provider_id == "aws-bedrock-pixtral"
    assert result.model_revision == "mistral.pixtral-large-2502-v1:0"
    assert result.claims[0].evidence_media_ids == ("MEDIA-WEB-1",)
    assert result.claims[0].surface_area.value_ha == 120
    assert result.claims[0].surface_area.upper_ha is None


def test_bedrock_pixtral_rejects_fabricated_media_reference() -> None:
    provider = BedrockPixtralMultimodalProvider(
        BedrockPixtralConfig(), client=_BedrockClient("MEDIA-INVENTED")
    )

    with pytest.raises(
        MultimodalEvidenceProviderError,
        match="multimodal_media_reference_rejected",
    ):
        provider.extract(_document(), allowed_claim_types=("area_burned",))


def test_bedrock_pixtral_requires_explicit_allowed_claim_policy() -> None:
    provider = BedrockPixtralMultimodalProvider(
        BedrockPixtralConfig(), client=_BedrockClient()
    )

    with pytest.raises(
        MultimodalEvidenceProviderError,
        match="multimodal_claim_policy_invalid",
    ):
        provider.extract(_document(), allowed_claim_types=())
