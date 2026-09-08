"""Dedicated-user service that launches one sandboxed Qwen process per research job."""

from __future__ import annotations

import json
import os
import socket
import socketserver
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from fireviewer_contracts.contracts import ResearchInputV1, ResearchOutputV1
from fireviewer_evidence_ingestion.research_rpc import call, read_message, write_message


class ResearchServiceError(RuntimeError):
    """Failure boundary between the brokered service and the sandboxed model."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        self.detail = detail or code
        super().__init__(self.detail)


def _broker_call(value: dict[str, Any]) -> dict[str, Any]:
    response = call(
        os.getenv("FW_RESEARCH_BROKER_SOCKET", "/run/firewarning/broker.sock"),
        value,
    )
    if response.get("ok") is not True or not isinstance(response.get("result"), dict):
        raise ResearchServiceError(
            "research_broker_request_failed",
            str(response.get("error") or "broker request failed"),
        )
    return dict(response["result"])


def _broker_policy(research: ResearchInputV1) -> dict[str, Any]:
    upload = research.private_upload
    return {
        "allowed_domains": list(research.allowed_domains),
        "search_templates": {
            domain: str(template) for domain, template in research.search_templates.items()
        },
        "max_fetch_bytes": research.max_fetch_bytes,
        "timeout_seconds": research.request_timeout_seconds,
        "pathname_prefix": upload.pathname_prefix,
        "upload_grant": upload.upload_grant,
        "token_endpoint": str(upload.token_endpoint),
        "resource_id": upload.resource_id,
        "maximum_file_size_bytes": upload.maximum_file_size_bytes,
        "allowed_content_types": list(upload.allowed_content_types),
    }


def _child_environment(*, session_token: str, temporary_home: Path) -> dict[str, str]:
    allowed_passthrough = (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "LD_LIBRARY_PATH",
    )
    environment = {key: os.environ[key] for key in allowed_passthrough if os.environ.get(key)}
    environment.update(
        {
            "HOME": str(temporary_home),
            "TMPDIR": str(temporary_home),
            "PATH": str(Path(sys.executable).parent),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HOME": os.getenv("HF_HOME", "/runpod-volume/huggingface-cache"),
            "FW_HF_CACHE_ROOT": os.getenv(
                "FW_HF_CACHE_ROOT", "/runpod-volume/huggingface-cache/hub"
            ),
            "FW_ATTENTION_IMPLEMENTATION": os.getenv(
                "FW_ATTENTION_IMPLEMENTATION", "flash_attention_2"
            ),
            "FW_RESEARCH_BROKER_SOCKET": os.getenv(
                "FW_RESEARCH_BROKER_SOCKET", "/run/firewarning/broker.sock"
            ),
            "FW_RESEARCH_SESSION_TOKEN": session_token,
        }
    )
    return environment


def execute_research(raw_input: object) -> dict[str, Any]:
    research = ResearchInputV1.model_validate(raw_input)
    control_token = os.getenv("FW_RESEARCH_BROKER_CONTROL_TOKEN", "")
    if len(control_token) < 32:
        raise ResearchServiceError(
            "research_broker_control_credential_unavailable",
            "research broker control credential is unavailable",
        )
    configured = _broker_call(
        {
            "action": "configure",
            "control_token": control_token,
            "policy": _broker_policy(research),
        }
    )
    session_token = str(configured.get("session_token", ""))
    if len(session_token) < 32:
        raise ResearchServiceError(
            "research_broker_invalid_session",
            "research broker returned an invalid session",
        )
    sanitized = research.model_dump(mode="json", exclude={"private_upload"})
    try:
        with tempfile.TemporaryDirectory(prefix="firewarning-research-") as directory:
            temporary_home = Path(directory)
            completed = subprocess.run(
                [sys.executable, "-m", "fireviewer_evidence_ingestion.research_agent"],
                input=json.dumps(sanitized, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=float(os.getenv("FW_RESEARCH_MODEL_TIMEOUT_SECONDS", "840")),
                check=False,
                env=_child_environment(
                    session_token=session_token,
                    temporary_home=temporary_home,
                ),
            )
        if completed.returncode != 0:
            detail = completed.stderr.strip()[-1_000:]
            raise ResearchServiceError(
                "sandboxed_research_process_failed",
                detail or "sandboxed research process exited without an error message",
            )
        output = ResearchOutputV1.model_validate_json(completed.stdout)
        if output.research_id != research.research_id:
            raise ResearchServiceError(
                "research_output_identifier_mismatch",
                "research output identifier does not match input",
            )
        return output.model_dump(mode="json")
    finally:
        _broker_call(
            {
                "action": "revoke",
                "control_token": control_token,
                "session_token": session_token,
            }
        )


class _ResearchHandler(socketserver.StreamRequestHandler):
    server: _ResearchServer

    def handle(self) -> None:
        try:
            request = read_message(self.rfile)
            if request.get("action") != "run":
                raise ResearchServiceError("research service action is invalid")
            output = execute_research(request.get("input"))
            write_message(self.wfile, {"ok": True, "output": output})
        except ResearchServiceError as exc:
            write_message(
                self.wfile,
                {"ok": False, "error": {"code": exc.code, "detail": exc.detail[:1_000]}},
            )
        except Exception as exc:
            write_message(
                self.wfile,
                {
                    "ok": False,
                    "error": {
                        "code": "research_service_unhandled_exception",
                        "detail": f"{type(exc).__name__}:{exc}"[:1_000],
                    },
                },
            )


class _UnixStreamServer(socketserver.TCPServer):
    address_family = getattr(socket, "AF_UNIX", socket.AF_INET)


class _ResearchServer(_UnixStreamServer):
    def __init__(self, path: str) -> None:
        if not hasattr(socket, "AF_UNIX"):
            raise RuntimeError("research service requires Linux AF_UNIX sockets")
        super().__init__(path, _ResearchHandler)  # type: ignore[arg-type]


def main() -> None:
    socket_path = Path(os.getenv("FW_RESEARCH_SERVICE_SOCKET", "/run/firewarning/research.sock"))
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    server = _ResearchServer(str(socket_path))
    os.chmod(socket_path, 0o660)
    server.serve_forever()


if __name__ == "__main__":
    main()
