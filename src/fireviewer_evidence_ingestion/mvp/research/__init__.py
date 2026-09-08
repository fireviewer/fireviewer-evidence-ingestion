"""Managed public-source research providers for the FireViewer MVP."""

from fireviewer_evidence_ingestion.mvp.research.agentcore_web_search import (
    AgentCoreWebSearchConfig,
    AgentCoreWebSearchProvider,
    McpToolClient,
)
from fireviewer_evidence_ingestion.mvp.research.multimodal_evidence import (
    AzureFederatedBedrockClient,
    AzureManagedIdentityWebTokenProvider,
    BedrockPixtralConfig,
    BedrockPixtralMultimodalProvider,
    ExtractedMultimodalClaim,
    MultimodalEvidenceDocument,
    MultimodalEvidenceExtraction,
    MultimodalEvidenceProvider,
    MultimodalEvidenceProviderError,
    TransientEvidenceImage,
)
from fireviewer_evidence_ingestion.mvp.research.source_acquisition import (
    CpuSourceAcquisitionWorker,
    SourceAcquisitionPlan,
    SourceAcquisitionRunReceipt,
    SourceDomainPolicy,
    build_source_acquisition_worker,
)
from fireviewer_evidence_ingestion.mvp.research.source_planner import (
    AutomaticSourceAcquisitionPlanner,
    AutomaticSourcePlannerConfig,
)

__all__ = [
    "AgentCoreWebSearchConfig",
    "AgentCoreWebSearchProvider",
    "AutomaticSourceAcquisitionPlanner",
    "AutomaticSourcePlannerConfig",
    "AzureFederatedBedrockClient",
    "AzureManagedIdentityWebTokenProvider",
    "BedrockPixtralConfig",
    "BedrockPixtralMultimodalProvider",
    "CpuSourceAcquisitionWorker",
    "ExtractedMultimodalClaim",
    "McpToolClient",
    "MultimodalEvidenceDocument",
    "MultimodalEvidenceExtraction",
    "MultimodalEvidenceProvider",
    "MultimodalEvidenceProviderError",
    "SourceAcquisitionPlan",
    "SourceAcquisitionRunReceipt",
    "SourceDomainPolicy",
    "TransientEvidenceImage",
    "build_source_acquisition_worker",
]
