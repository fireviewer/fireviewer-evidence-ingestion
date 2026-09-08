"""Network broker for the sandboxed source-research model process."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import mimetypes
import os
import secrets
import socket
import socketserver
import tempfile
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlsplit

import httpx

from fireviewer_evidence_ingestion.research_rpc import read_message, write_message


class BrokerPolicyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BrokerPolicy:
    allowed_domains: frozenset[str]
    search_provider_domains: frozenset[str]
    search_templates: dict[str, str]
    max_fetch_bytes: int
    max_media_fetch_bytes: int
    timeout_seconds: int
    pathname_prefix: str | None
    upload_grant: str | None
    token_endpoint: str | None
    resource_id: str | None
    maximum_file_size_bytes: int | None
    allowed_content_types: frozenset[str]


class _LinkParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[dict[str, str]] = []
        self.media_links: list[str] = []
        self.metadata: dict[str, str] = {}
        self._href: str | None = None
        self._text: list[str] = []

    def _media(self, value: str | None) -> None:
        if value:
            self.media_links.append(urljoin(self.base_url, value))

    def _srcset(self, value: str | None) -> None:
        if not value:
            return
        for candidate in value.split(","):
            source = candidate.strip().split(" ", 1)[0]
            self._media(source)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag.casefold()
        values = {key.casefold(): value for key, value in attrs if value is not None}
        if normalized_tag == "meta":
            key = (values.get("property") or values.get("name") or "").casefold()
            content = values.get("content")
            if key and content:
                self.metadata.setdefault(key, content)
                if key in {
                    "og:image",
                    "og:image:url",
                    "og:image:secure_url",
                    "twitter:image",
                    "twitter:image:src",
                    "og:video",
                    "og:video:url",
                    "og:video:secure_url",
                    "twitter:player:stream",
                    "og:audio",
                    "og:audio:url",
                    "og:audio:secure_url",
                }:
                    self._media(content)
            return
        if normalized_tag in {"video", "audio", "source"}:
            self._media(values.get("src"))
            if normalized_tag == "video":
                self._media(values.get("poster"))
            return
        if normalized_tag == "img":
            source = (
                values.get("data-src")
                or values.get("data-lazy-src")
                or values.get("data-original")
                or values.get("src")
            )
            if source:
                self._media(source)
            self._srcset(values.get("data-srcset") or values.get("srcset"))
            return
        if normalized_tag == "link" and (
            values.get("rel", "").casefold() in {"image_src", "preload"}
        ):
            self._media(values.get("href"))
            return
        if normalized_tag != "a":
            return
        href = values.get("href")
        if href:
            path = urlsplit(href).path.casefold()
            if path.endswith(
                (
                    ".avif",
                    ".jpeg",
                    ".jpg",
                    ".png",
                    ".webp",
                    ".m4a",
                    ".mp3",
                    ".mp4",
                    ".mov",
                    ".wav",
                    ".webm",
                )
            ):
                self._media(href)
            self._href = urljoin(self.base_url, href)
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href:
            normalized = " ".join(data.split())
            if normalized:
                self._text.append(normalized)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "a" and self._href:
            self.links.append({"url": self._href, "title": " ".join(self._text)[:500]})
            self._href = None
            self._text = []


class ResearchBroker:
    def __init__(
        self,
        *,
        control_token: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.control_token = control_token
        self._transport = transport
        self._sessions: dict[str, BrokerPolicy] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _domain_allowed(host: str, allowed: frozenset[str]) -> bool:
        return any(host == domain or host.endswith(f".{domain}") for domain in allowed)

    @staticmethod
    def _public_addresses(host: str, port: int) -> tuple[str, ...]:
        try:
            results = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise BrokerPolicyError("broker_dns_resolution_failed") from exc
        addresses = tuple(sorted({str(item[4][0]) for item in results}))
        if not addresses:
            raise BrokerPolicyError("broker_dns_resolution_empty")
        for value in addresses:
            address = ipaddress.ip_address(value)
            if not address.is_global:
                raise BrokerPolicyError("broker_private_address_forbidden")
        return addresses

    def _guard_url(self, value: str, *, allowed_domains: frozenset[str]) -> str:
        parts = urlsplit(value)
        host = (parts.hostname or "").casefold().rstrip(".")
        if (
            parts.scheme.casefold() != "https"
            or not host
            or parts.username is not None
            or parts.password is not None
            or parts.fragment
            or parts.port not in {None, 443}
        ):
            raise BrokerPolicyError("broker_https_url_required")
        if not self._domain_allowed(host, allowed_domains):
            raise BrokerPolicyError("broker_domain_forbidden")
        self._public_addresses(host, 443)
        return value

    @staticmethod
    def _policy(value: dict[str, Any]) -> BrokerPolicy:
        allowed = frozenset(
            str(domain).strip().casefold().rstrip(".")
            for domain in value.get("allowed_domains", [])
            if str(domain).strip()
        )
        templates = {
            str(domain).strip().casefold().rstrip("."): str(template)
            for domain, template in dict(value.get("search_templates", {})).items()
        }
        providers = frozenset(templates)
        if not allowed or not providers or providers & allowed:
            raise BrokerPolicyError("broker_policy_domains_invalid")
        for provider, template in templates.items():
            parts = urlsplit(template)
            host = (parts.hostname or "").casefold().rstrip(".")
            if (
                parts.scheme.casefold() != "https"
                or host != provider
                or parts.username is not None
                or parts.password is not None
                or parts.fragment
                or parts.port not in {None, 443}
                or template.count("{query}") != 1
            ):
                raise BrokerPolicyError("broker_search_template_invalid")
        max_fetch_bytes = int(value.get("max_fetch_bytes", 0))
        max_media_fetch_bytes = int(value.get("max_media_fetch_bytes", max_fetch_bytes))
        timeout_seconds = int(value.get("timeout_seconds", 0))
        if not 65_536 <= max_fetch_bytes <= 512 * 1_024 * 1_024:
            raise BrokerPolicyError("broker_policy_fetch_limit_invalid")
        if not max_fetch_bytes <= max_media_fetch_bytes <= 512 * 1_024 * 1_024:
            raise BrokerPolicyError("broker_policy_media_fetch_limit_invalid")
        if not 2 <= timeout_seconds <= 120:
            raise BrokerPolicyError("broker_policy_timeout_invalid")
        upload_keys = {
            "pathname_prefix",
            "upload_grant",
            "token_endpoint",
            "resource_id",
            "maximum_file_size_bytes",
            "allowed_content_types",
        }
        supplied_upload_keys = upload_keys & set(value)
        if supplied_upload_keys and supplied_upload_keys != upload_keys:
            raise BrokerPolicyError("broker_policy_upload_incomplete")
        if supplied_upload_keys:
            maximum_file_size_bytes = int(value["maximum_file_size_bytes"])
            if not 1_048_576 <= maximum_file_size_bytes <= 1_073_741_824:
                raise BrokerPolicyError("broker_policy_upload_limit_invalid")
            pathname_prefix = str(value["pathname_prefix"])
            upload_grant = str(value["upload_grant"])
            token_endpoint = str(value["token_endpoint"])
            resource_id = str(value["resource_id"])
            allowed_content_types = frozenset(
                str(item) for item in value["allowed_content_types"]
            )
        else:
            maximum_file_size_bytes = None
            pathname_prefix = None
            upload_grant = None
            token_endpoint = None
            resource_id = None
            allowed_content_types = frozenset()
        return BrokerPolicy(
            allowed_domains=allowed,
            search_provider_domains=providers,
            search_templates=templates,
            max_fetch_bytes=max_fetch_bytes,
            max_media_fetch_bytes=max_media_fetch_bytes,
            timeout_seconds=timeout_seconds,
            pathname_prefix=pathname_prefix,
            upload_grant=upload_grant,
            token_endpoint=token_endpoint,
            resource_id=resource_id,
            maximum_file_size_bytes=maximum_file_size_bytes,
            allowed_content_types=allowed_content_types,
        )

    def configure(self, request: dict[str, Any]) -> dict[str, Any]:
        supplied = str(request.get("control_token", ""))
        if not hmac.compare_digest(supplied, self.control_token):
            raise BrokerPolicyError("broker_control_unauthorized")
        policy = self._policy(dict(request.get("policy", {})))
        session_token = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[session_token] = policy
        return {"session_token": session_token}

    def revoke(self, request: dict[str, Any]) -> dict[str, Any]:
        supplied = str(request.get("control_token", ""))
        if not hmac.compare_digest(supplied, self.control_token):
            raise BrokerPolicyError("broker_control_unauthorized")
        with self._lock:
            self._sessions.pop(str(request.get("session_token", "")), None)
        return {"revoked": True}

    def _session(self, request: dict[str, Any]) -> BrokerPolicy:
        token = str(request.get("session_token", ""))
        with self._lock:
            policy = self._sessions.get(token)
        if policy is None:
            raise BrokerPolicyError("broker_session_unauthorized")
        return policy

    def _request(
        self,
        method: str,
        url: str,
        *,
        policy: BrokerPolicy,
        allowed_domains: frozenset[str],
    ) -> tuple[httpx.Response, bytes]:
        guarded = self._guard_url(url, allowed_domains=allowed_domains)
        headers = {
            "User-Agent": "FireWarning-SourceResearch/1.0 (+private-human-review)",
            "Accept": "text/html,application/json,text/plain,image/*,audio/*,video/*",
        }
        with (
            httpx.Client(
                timeout=policy.timeout_seconds,
                follow_redirects=False,
                headers=headers,
                transport=self._transport,
            ) as client,
            client.stream(method, guarded) as response,
        ):
            if 300 <= response.status_code < 400:
                raise BrokerPolicyError("broker_redirect_forbidden")
            response.raise_for_status()
            if method == "HEAD":
                return response, b""
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > policy.max_fetch_bytes:
                    raise BrokerPolicyError("broker_response_too_large")
                chunks.append(chunk)
            return response, b"".join(chunks)

    @staticmethod
    def _response_metadata(response: httpx.Response) -> dict[str, Any]:
        return {
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type", "").split(";", 1)[0],
            "content_length": response.headers.get("content-length"),
            "last_modified": response.headers.get("last-modified"),
            "etag": response.headers.get("etag"),
            "url": str(response.url),
            "retrieved_at": datetime.now(UTC).isoformat(),
        }

    def search(self, request: dict[str, Any], policy: BrokerPolicy) -> dict[str, Any]:
        arguments = dict(request.get("arguments", {}))
        domain = str(arguments.get("domain", "")).casefold().rstrip(".")
        query = str(arguments.get("query", "")).strip()
        try:
            offset = int(arguments.get("cursor", 0) or 0)
            limit = int(arguments.get("limit", 20))
        except (TypeError, ValueError) as exc:
            raise BrokerPolicyError("broker_search_pagination_invalid") from exc
        if offset < 0 or not 1 <= limit <= 50:
            raise BrokerPolicyError("broker_search_pagination_invalid")
        template = policy.search_templates.get(domain)
        if template is None or not query or "{query}" not in template:
            raise BrokerPolicyError("broker_search_request_invalid")
        url = template.replace("{query}", quote_plus(query))
        response, content = self._request(
            "GET",
            url,
            policy=policy,
            allowed_domains=policy.search_provider_domains,
        )
        parser = _LinkParser(str(response.url))
        parser.feed(content.decode("utf-8", errors="replace"))
        all_links: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for link in parser.links:
            try:
                guarded = self._source_link(link["url"], policy=policy)
            except BrokerPolicyError:
                continue
            if guarded in seen_urls:
                continue
            seen_urls.add(guarded)
            all_links.append({"url": guarded, "title": link["title"]})
            if len(all_links) >= 500:
                break
        links = all_links[offset : offset + limit]
        next_cursor = str(offset + len(links)) if offset + len(links) < len(all_links) else None
        return {
            "query": query,
            "domain": domain,
            "links": links,
            "cursor": str(offset) if offset else None,
            "next_cursor": next_cursor,
        }

    def _source_link(self, value: str, *, policy: BrokerPolicy) -> str:
        try:
            return self._guard_url(value, allowed_domains=policy.allowed_domains)
        except BrokerPolicyError as direct_error:
            parts = urlsplit(value)
            provider_host = (parts.hostname or "").casefold().rstrip(".")
            if not self._domain_allowed(provider_host, policy.search_provider_domains):
                raise direct_error
            query = parse_qs(parts.query)
            for key in ("uddg", "url", "u"):
                candidates = query.get(key, [])
                if not candidates:
                    continue
                candidate = unquote(str(candidates[0]))
                try:
                    return self._guard_url(
                        candidate,
                        allowed_domains=policy.allowed_domains,
                    )
                except BrokerPolicyError:
                    continue
            raise direct_error

    def inspect(self, request: dict[str, Any], policy: BrokerPolicy) -> dict[str, Any]:
        url = str(dict(request.get("arguments", {})).get("url", ""))
        response, _content = self._request(
            "HEAD",
            url,
            policy=policy,
            allowed_domains=policy.allowed_domains,
        )
        return self._response_metadata(response)

    def _upload(
        self,
        *,
        content: bytes,
        content_type: str,
        pathname: str,
        policy: BrokerPolicy,
    ) -> dict[str, Any]:
        if (
            policy.pathname_prefix is None
            or policy.upload_grant is None
            or policy.token_endpoint is None
            or policy.resource_id is None
            or policy.maximum_file_size_bytes is None
        ):
            raise BrokerPolicyError("broker_upload_disabled")
        from vercel.blob import BlobClient

        if content_type not in policy.allowed_content_types:
            raise BrokerPolicyError("broker_upload_content_type_forbidden")
        if len(content) > policy.maximum_file_size_bytes:
            raise BrokerPolicyError("broker_upload_too_large")
        if not pathname.startswith(f"{policy.pathname_prefix}/"):
            raise BrokerPolicyError("broker_upload_path_forbidden")
        token_parts = urlsplit(policy.token_endpoint)
        token_host = (token_parts.hostname or "").casefold().rstrip(".")
        if (
            token_parts.scheme.casefold() != "https"
            or not token_host
            or token_parts.username is not None
            or token_parts.password is not None
            or token_parts.fragment
            or token_parts.port not in {None, 443}
        ):
            raise BrokerPolicyError("broker_upload_token_url_invalid")
        self._public_addresses(token_host, 443)
        with httpx.Client(
            timeout=policy.timeout_seconds,
            follow_redirects=False,
            transport=self._transport,
        ) as client:
            token_response = client.post(
                policy.token_endpoint,
                headers={"X-Blob-Upload-Grant": policy.upload_grant},
                json={
                    "type": "blob.generate-client-token",
                    "payload": {
                        "pathname": pathname,
                        "multipart": True,
                        "clientPayload": policy.resource_id,
                    },
                },
            )
        if 300 <= token_response.status_code < 400:
            raise BrokerPolicyError("broker_upload_token_redirect_forbidden")
        token_response.raise_for_status()
        client_token = token_response.json().get("clientToken")
        if not isinstance(client_token, str) or not client_token:
            raise BrokerPolicyError("broker_upload_token_invalid")
        suffix = mimetypes.guess_extension(content_type) or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as stream:
            stream.write(content)
            temp_path = Path(stream.name)
        try:
            with BlobClient(token=client_token) as client:
                result = client.upload_file(
                    temp_path,
                    pathname,
                    access="private",
                    content_type=content_type,
                    add_random_suffix=False,
                    overwrite=False,
                    multipart=True,
                    cache_control_max_age=31_536_000,
                )
        finally:
            temp_path.unlink(missing_ok=True)
        if result.pathname != pathname:
            raise BrokerPolicyError("broker_upload_path_mismatch")
        return {
            "blob_pathname": result.pathname,
            "media_sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
            "content_type": content_type,
        }

    def _fetch_public_content(
        self,
        *,
        url: str,
        policy: BrokerPolicy,
    ) -> tuple[dict[str, Any], bytes]:
        response, content = self._request(
            "GET",
            url,
            policy=policy,
            allowed_domains=policy.allowed_domains,
        )
        metadata = self._response_metadata(response)
        content_type = str(metadata["content_type"])
        result: dict[str, Any] = {**metadata, "sha256": hashlib.sha256(content).hexdigest()}
        result["size_bytes"] = len(content)
        if content_type.startswith(("text/", "application/json")):
            text = content.decode("utf-8", errors="replace")[:100_000]
            result["text"] = text
            if content_type == "text/html":
                parser = _LinkParser(url)
                parser.feed(text)
                result["links"] = parser.links[:50]
                result["media_links"] = list(dict.fromkeys(parser.media_links))[:100]
                result["metadata"] = parser.metadata
        return result, content

    def fetch_transient_bytes(
        self,
        request: dict[str, Any],
        policy: BrokerPolicy,
    ) -> tuple[dict[str, Any], bytes]:
        """Return one allowlisted public object in memory without any storage action."""

        arguments = dict(request.get("arguments", {}))
        if bool(arguments.get("store", False)):
            raise BrokerPolicyError("broker_transient_store_forbidden")
        result, content = self.fetch_transient_media(request, policy)
        if content is None:
            raise BrokerPolicyError("broker_transient_memory_limit_exceeded")
        return result, content

    def fetch_transient_media(
        self,
        request: dict[str, Any],
        policy: BrokerPolicy,
        *,
        maximum_in_memory_bytes: int = 8 * 1_024 * 1_024,
    ) -> tuple[dict[str, Any], bytes | None]:
        """Hash public media in a stream; retain bytes only for small image inputs."""

        arguments = dict(request.get("arguments", {}))
        if bool(arguments.get("store", False)):
            raise BrokerPolicyError("broker_transient_store_forbidden")
        if not 0 <= maximum_in_memory_bytes <= 16 * 1_024 * 1_024:
            raise BrokerPolicyError("broker_transient_memory_limit_invalid")
        url = self._guard_url(
            str(arguments.get("url", "")),
            allowed_domains=policy.allowed_domains,
        )
        headers = {
            "User-Agent": "FireWarning-SourceResearch/1.0 (+private-human-review)",
            "Accept": "image/*,audio/*,video/*",
        }
        digest = hashlib.sha256()
        size = 0
        buffered: list[bytes] | None = []
        with (
            httpx.Client(
                timeout=policy.timeout_seconds,
                follow_redirects=False,
                headers=headers,
                transport=self._transport,
            ) as client,
            client.stream("GET", url) as response,
        ):
            if 300 <= response.status_code < 400:
                raise BrokerPolicyError("broker_redirect_forbidden")
            response.raise_for_status()
            metadata = self._response_metadata(response)
            content_type = str(metadata["content_type"])
            retain = content_type.startswith("image/") and maximum_in_memory_bytes > 0
            if not retain:
                buffered = None
            declared = response.headers.get("content-length")
            if (
                declared
                and declared.isdecimal()
                and int(declared) > policy.max_media_fetch_bytes
            ):
                raise BrokerPolicyError("broker_response_too_large")
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > policy.max_media_fetch_bytes:
                    raise BrokerPolicyError("broker_response_too_large")
                digest.update(chunk)
                if buffered is not None:
                    if size <= maximum_in_memory_bytes:
                        buffered.append(chunk)
                    else:
                        buffered = None
        result = {
            **metadata,
            "sha256": digest.hexdigest(),
            "size_bytes": size,
            "binary_stored": False,
        }
        return result, b"".join(buffered) if buffered is not None else None

    def materialize_transient_file(
        self,
        request: dict[str, Any],
        policy: BrokerPolicy,
        *,
        destination: Path,
        expected_sha256: str,
        expected_size_bytes: int,
        expected_content_type: str,
    ) -> dict[str, Any]:
        """Stream one verified public object to an ephemeral caller-owned path."""

        arguments = dict(request.get("arguments", {}))
        if bool(arguments.get("store", False)):
            raise BrokerPolicyError("broker_transient_store_forbidden")
        if (
            not expected_size_bytes > 0
            or expected_size_bytes > policy.max_media_fetch_bytes
            or len(expected_sha256) != 64
        ):
            raise BrokerPolicyError("broker_transient_expectation_invalid")
        url = self._guard_url(
            str(arguments.get("url", "")),
            allowed_domains=policy.allowed_domains,
        )
        if destination.exists() or not destination.parent.is_dir():
            raise BrokerPolicyError("broker_transient_destination_invalid")
        headers = {
            "User-Agent": "FireWarning-PublicMedia/1.0 (+private-human-review)",
            "Accept": expected_content_type,
        }
        digest = hashlib.sha256()
        size = 0
        try:
            with (
                httpx.Client(
                    timeout=policy.timeout_seconds,
                    follow_redirects=False,
                    headers=headers,
                    transport=self._transport,
                ) as client,
                client.stream("GET", url) as response,
            ):
                if 300 <= response.status_code < 400:
                    raise BrokerPolicyError("broker_redirect_forbidden")
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";", 1)[0]
                if content_type != expected_content_type:
                    raise BrokerPolicyError("broker_transient_content_type_mismatch")
                declared = response.headers.get("content-length")
                if declared and declared.isdecimal() and int(declared) != expected_size_bytes:
                    raise BrokerPolicyError("broker_transient_size_mismatch")
                with destination.open("xb") as stream:
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > policy.max_media_fetch_bytes or size > expected_size_bytes:
                            raise BrokerPolicyError("broker_response_too_large")
                        digest.update(chunk)
                        stream.write(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        if size != expected_size_bytes:
            destination.unlink(missing_ok=True)
            raise BrokerPolicyError("broker_transient_size_mismatch")
        if digest.hexdigest() != expected_sha256:
            destination.unlink(missing_ok=True)
            raise BrokerPolicyError("broker_transient_digest_mismatch")
        return {
            "url": url,
            "content_type": expected_content_type,
            "size_bytes": size,
            "sha256": expected_sha256,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "binary_stored": False,
        }

    def fetch(self, request: dict[str, Any], policy: BrokerPolicy) -> dict[str, Any]:
        arguments = dict(request.get("arguments", {}))
        url = str(arguments.get("url", ""))
        result, content = self._fetch_public_content(url=url, policy=policy)
        content_type = str(result["content_type"])
        if bool(arguments.get("store", False)):
            if policy.pathname_prefix is None:
                raise BrokerPolicyError("broker_upload_disabled")
            candidate_id = str(arguments.get("candidate_id", ""))
            suffix = Path(urlsplit(url).path).suffix.casefold()
            if not candidate_id or not suffix or len(suffix) > 10:
                raise BrokerPolicyError("broker_store_request_invalid")
            pathname = f"{policy.pathname_prefix}/{candidate_id}{suffix}"
            result.update(
                self._upload(
                    content=content,
                    content_type=content_type,
                    pathname=pathname,
                    policy=policy,
                )
            )
        return result

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        action = request.get("action")
        if action == "configure":
            return self.configure(request)
        if action == "revoke":
            return self.revoke(request)
        policy = self._session(request)
        if action == "search":
            return self.search(request, policy)
        if action == "fetch":
            return self.fetch(request, policy)
        if action == "inspect":
            return self.inspect(request, policy)
        raise BrokerPolicyError("broker_action_invalid")


class _BrokerHandler(socketserver.StreamRequestHandler):
    server: _BrokerServer

    def handle(self) -> None:
        try:
            request = read_message(self.rfile)
            result = self.server.broker.handle(request)
            write_message(self.wfile, {"ok": True, "result": result})
        except Exception as exc:
            write_message(
                self.wfile,
                {"ok": False, "error": f"{type(exc).__name__}:{exc}"[:1_000]},
            )


class _UnixStreamServer(socketserver.TCPServer):
    address_family = getattr(socket, "AF_UNIX", socket.AF_INET)


class _BrokerServer(socketserver.ThreadingMixIn, _UnixStreamServer):
    daemon_threads = True

    def __init__(self, path: str, *, broker: ResearchBroker) -> None:
        if not hasattr(socket, "AF_UNIX"):
            raise RuntimeError("research broker requires Linux AF_UNIX sockets")
        self.broker = broker
        super().__init__(path, _BrokerHandler)  # type: ignore[arg-type]


def main() -> None:
    socket_path = Path(os.getenv("FW_RESEARCH_BROKER_SOCKET", "/run/firewarning/broker.sock"))
    control_token = os.getenv("FW_RESEARCH_BROKER_CONTROL_TOKEN", "")
    if len(control_token) < 32:
        raise SystemExit("FW_RESEARCH_BROKER_CONTROL_TOKEN must contain 32 characters")
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    server = _BrokerServer(str(socket_path), broker=ResearchBroker(control_token=control_token))
    os.chmod(socket_path, 0o660)
    server.serve_forever()


if __name__ == "__main__":
    main()
