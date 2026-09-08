from __future__ import annotations

import pytest

from fireviewer_evidence_ingestion.research_agent import _build_candidates

SOURCE_POLICIES = {
    "sources.example": {
        "source_name": "Source de test",
        "kind": "authority",
        "scope": "local",
        "confidence_level": "A+",
        "claim_types": ["operational_confirmation"],
        "publication_policy": "per_item_license_check",
        "minimum_refresh_minutes": 10,
    }
}


def test_final_candidate_must_have_been_fetched_by_broker() -> None:
    with pytest.raises(ValueError, match="not fetched"):
        _build_candidates(
            {
                "candidates": [
                    {
                        "url": "https://sources.example/fire",
                        "media_type": "article",
                    }
                ]
            },
            fetched={},
            source_policies=SOURCE_POLICIES,
        )


def test_media_hash_and_private_path_come_from_broker_evidence() -> None:
    url = "https://sources.example/fire.jpg"
    candidates = _build_candidates(
        {
            "candidates": [
                {
                    "url": url,
                    "title": "Photo du feu",
                    "published_at": "2026-07-09T08:00:00+02:00",
                    "media_type": "image",
                    "blob_pathname": "model-must-not-control-this-value",
                    "media_sha256": "0" * 64,
                    "size_bytes": 1,
                }
            ]
        },
        fetched={
            url: {
                "url": url,
                "content_type": "image/jpeg",
                "retrieved_at": "2026-07-09T09:00:00+02:00",
                "sha256": "a" * 64,
                "blob_pathname": "firewarning/source-packages/upload/candidate.jpg",
                "media_sha256": "a" * 64,
                "size_bytes": 4096,
            }
        },
        source_policies=SOURCE_POLICIES,
    )

    assert candidates[0]["blob_pathname"].startswith("firewarning/source-packages/")
    assert candidates[0]["media_sha256"] == "a" * 64
    assert candidates[0]["size_bytes"] == 4096
    assert candidates[0]["provenance"]["source_policy"]["confidence_level"] == "A+"
