"""Worker-side client for the isolated source-research service."""

from __future__ import annotations

import os
from pathlib import Path

from fireviewer_contracts.contracts import ResearchInputV1, ResearchOutputV1
from fireviewer_evidence_ingestion.research_rpc import call


class ResearchServiceError(RuntimeError):
    """A structured failure reported by the isolated research service."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        self.detail = detail or code
        super().__init__(self.detail)


def run_isolated_research(research: ResearchInputV1) -> ResearchOutputV1:
    socket_path = Path(os.getenv("FW_RESEARCH_SERVICE_SOCKET", "/run/firewarning/research.sock"))
    response = call(
        socket_path,
        {"action": "run", "input": research.model_dump(mode="json")},
        timeout=float(os.getenv("FW_RESEARCH_SERVICE_TIMEOUT_SECONDS", "900")),
    )
    if response.get("ok") is not True:
        error = response.get("error")
        if isinstance(error, dict):
            code = str(error.get("code") or "research_service_failed")
            detail = str(error.get("detail") or code)
        else:
            code = "research_service_failed"
            detail = str(error or code)
        raise ResearchServiceError(code, detail)
    output = response.get("output")
    return ResearchOutputV1.model_validate(output)
