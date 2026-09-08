from __future__ import annotations

import socket
from hashlib import sha256
from pathlib import Path

import httpx
import pytest

from fireviewer_evidence_ingestion.research_broker import BrokerPolicyError, ResearchBroker

CONTROL_TOKEN = "control-token-for-tests-0000000000000000"  # noqa: S105


def _configure(broker: ResearchBroker) -> str:
    result = broker.configure(
        {
            "control_token": CONTROL_TOKEN,
            "policy": {
                "allowed_domains": ["sources.example"],
                "search_templates": {"search.example": "https://search.example/search?q={query}"},
                "max_fetch_bytes": 65_536,
                "timeout_seconds": 5,
                "pathname_prefix": "firewarning/source-packages/upload-test",
                "upload_grant": "g" * 128,
                "token_endpoint": "https://backend.example/api/v1/admin/blob-upload-token",
                "resource_id": "research-test-0001",
                "maximum_file_size_bytes": 1_048_576,
                "allowed_content_types": ["image/jpeg", "text/html"],
            },
        }
    )
    return str(result["session_token"])


def _public_dns(*_args, **_kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]


def test_broker_rejects_domain_outside_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    broker = ResearchBroker(control_token=CONTROL_TOKEN)
    token = _configure(broker)

    with pytest.raises(BrokerPolicyError, match="broker_domain_forbidden"):
        broker.fetch(
            {
                "session_token": token,
                "arguments": {"url": "https://untrusted.example/fire.jpg"},
            },
            broker._session({"session_token": token}),
        )


def test_search_provider_cannot_be_fetched_as_a_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    broker = ResearchBroker(control_token=CONTROL_TOKEN)
    token = _configure(broker)

    with pytest.raises(BrokerPolicyError, match="broker_domain_forbidden"):
        broker.fetch(
            {
                "session_token": token,
                "arguments": {"url": "https://search.example/result-page"},
            },
            broker._session({"session_token": token}),
        )


def test_search_returns_only_allowlisted_source_links(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    html = b"""
    <a href="https://sources.example/fire">trusted</a>
    <a href="https://untrusted.example/fire">untrusted</a>
    <a href="https://search.example/?uddg=https%3A%2F%2Fsources.example%2Fphoto">redirect</a>
    """
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=html,
            request=request,
        )
    )
    broker = ResearchBroker(control_token=CONTROL_TOKEN, transport=transport)
    token = _configure(broker)
    policy = broker._session({"session_token": token})

    result = broker.search(
        {
            "session_token": token,
            "arguments": {"domain": "search.example", "query": "feu de démonstration"},
        },
        policy,
    )

    assert [link["url"] for link in result["links"]] == [
        "https://sources.example/fire",
        "https://sources.example/photo",
    ]


def test_search_paginates_and_fetch_exposes_media_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "search.example":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=b"""
                <a href="https://sources.example/one">one</a>
                <a href="https://sources.example/two">two</a>
                <a href="https://sources.example/three">three</a>
                """,
                request=request,
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"""
            <meta property="og:title" content="Incident update">
            <meta property="og:image" content="/media/fire.jpg">
            <meta property="og:video:secure_url" content="/media/briefing.mp4">
            <video poster="/media/poster.jpg"><source src="/media/drone.webm"></video>
            <audio src="/media/briefing.mp3"></audio>
            <img src="https://sources.example/media/smoke.jpg">
            """,
            request=request,
        )

    broker = ResearchBroker(
        control_token=CONTROL_TOKEN,
        transport=httpx.MockTransport(handler),
    )
    token = _configure(broker)
    policy = broker._session({"session_token": token})
    first = broker.search(
        {
            "arguments": {
                "domain": "search.example",
                "query": "fire",
                "limit": 2,
            }
        },
        policy,
    )
    second = broker.search(
        {
            "arguments": {
                "domain": "search.example",
                "query": "fire",
                "cursor": first["next_cursor"],
                "limit": 2,
            }
        },
        policy,
    )
    fetched = broker.fetch(
        {"arguments": {"url": "https://sources.example/one"}},
        policy,
    )

    assert [item["url"] for item in first["links"]] == [
        "https://sources.example/one",
        "https://sources.example/two",
    ]
    assert first["next_cursor"] == "2"
    assert [item["url"] for item in second["links"]] == [
        "https://sources.example/three"
    ]
    assert second["next_cursor"] is None
    assert fetched["size_bytes"] > 0
    assert fetched["metadata"]["og:title"] == "Incident update"
    assert fetched["media_links"] == [
        "https://sources.example/media/fire.jpg",
        "https://sources.example/media/briefing.mp4",
        "https://sources.example/media/poster.jpg",
        "https://sources.example/media/drone.webm",
        "https://sources.example/media/briefing.mp3",
        "https://sources.example/media/smoke.jpg",
    ]


def test_broker_rejects_private_dns_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    )
    broker = ResearchBroker(control_token=CONTROL_TOKEN)
    token = _configure(broker)

    with pytest.raises(BrokerPolicyError, match="broker_private_address_forbidden"):
        broker.inspect(
            {
                "session_token": token,
                "arguments": {"url": "https://sources.example/fire"},
            },
            broker._session({"session_token": token}),
        )


def test_broker_refuses_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            302,
            headers={"location": "https://sources.example/other"},
            request=request,
        )
    )
    broker = ResearchBroker(control_token=CONTROL_TOKEN, transport=transport)
    token = _configure(broker)

    with pytest.raises(BrokerPolicyError, match="broker_redirect_forbidden"):
        broker.fetch(
            {
                "session_token": token,
                "arguments": {"url": "https://sources.example/fire"},
            },
            broker._session({"session_token": token}),
        )


def test_broker_enforces_streamed_response_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"x" * 65_537,
            request=request,
        )
    )
    broker = ResearchBroker(control_token=CONTROL_TOKEN, transport=transport)
    token = _configure(broker)

    with pytest.raises(BrokerPolicyError, match="broker_response_too_large"):
        broker.fetch(
            {
                "session_token": token,
                "arguments": {"url": "https://sources.example/fire"},
            },
            broker._session({"session_token": token}),
        )


def test_revoked_session_cannot_use_network_tools() -> None:
    broker = ResearchBroker(control_token=CONTROL_TOKEN)
    token = _configure(broker)
    broker.revoke({"control_token": CONTROL_TOKEN, "session_token": token})

    with pytest.raises(BrokerPolicyError, match="broker_session_unauthorized"):
        broker.handle(
            {
                "action": "inspect",
                "session_token": token,
                "arguments": {"url": "https://sources.example/fire"},
            }
        )


def test_fetch_only_policy_has_no_vercel_upload_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    broker = ResearchBroker(
        control_token=CONTROL_TOKEN,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=b"source metadata only",
                request=request,
            )
        ),
    )
    configured = broker.configure(
        {
            "control_token": CONTROL_TOKEN,
            "policy": {
                "allowed_domains": ["sources.example"],
                "search_templates": {
                    "search.example": "https://search.example/search?q={query}"
                },
                "max_fetch_bytes": 65_536,
                "timeout_seconds": 5,
            },
        }
    )
    policy = broker._session({"session_token": configured["session_token"]})

    fetched = broker.fetch(
        {"arguments": {"url": "https://sources.example/fire"}},
        policy,
    )
    assert fetched["sha256"]

    with pytest.raises(BrokerPolicyError, match="broker_upload_disabled"):
        broker.fetch(
            {
                "arguments": {
                    "url": "https://sources.example/fire.jpg",
                    "store": True,
                    "candidate_id": "candidate-1",
                }
            },
            policy,
        )


def test_public_media_is_streamed_to_ephemeral_file_with_exact_ticket(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    video = b"verified-public-video"
    broker = ResearchBroker(
        control_token=CONTROL_TOKEN,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={
                    "content-type": "video/mp4",
                    "content-length": str(len(video)),
                },
                content=video,
                request=request,
            )
        ),
    )
    token = _configure(broker)
    policy = broker._session({"session_token": token})
    destination = tmp_path / "public-video.bin"

    receipt = broker.materialize_transient_file(
        {"arguments": {"url": "https://sources.example/point.mp4"}},
        policy,
        destination=destination,
        expected_sha256=sha256(video).hexdigest(),
        expected_size_bytes=len(video),
        expected_content_type="video/mp4",
    )

    assert destination.read_bytes() == video
    assert receipt["binary_stored"] is False
    destination.unlink()
    assert not destination.exists()


def test_public_media_digest_failure_removes_ephemeral_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    video = b"tampered-public-video"
    broker = ResearchBroker(
        control_token=CONTROL_TOKEN,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "video/mp4"},
                content=video,
                request=request,
            )
        ),
    )
    token = _configure(broker)
    policy = broker._session({"session_token": token})
    destination = tmp_path / "rejected-video.bin"

    with pytest.raises(BrokerPolicyError, match="broker_transient_digest_mismatch"):
        broker.materialize_transient_file(
            {"arguments": {"url": "https://sources.example/point.mp4"}},
            policy,
            destination=destination,
            expected_sha256="0" * 64,
            expected_size_bytes=len(video),
            expected_content_type="video/mp4",
        )

    assert not destination.exists()
