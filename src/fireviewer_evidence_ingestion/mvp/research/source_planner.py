"""Deterministic source-acquisition plans derived from durable candidates."""

from __future__ import annotations

import json
import re
from hashlib import sha256

from pydantic import Field, model_validator

from fireviewer_contracts.contracts import StrictModel
from fireviewer_evidence_ingestion.mvp.research.source_acquisition import (
    SourceAcquisitionPlan,
    SourceDomainPolicy,
)
from fireviewer_contracts.mvp.supervision.backend_event_evidence import DurableEventEvidence

_UNSAFE_LABEL_PATTERN = re.compile(r"(?:https?://|@|\+?\d[\d .()-]{7,}\d)", re.IGNORECASE)
_LABEL_CHARACTER_PATTERN = re.compile(r"[^\wÀ-ÖØ-öø-ÿ' -]+", re.UNICODE)
_FOCUS_PATTERN = re.compile(r"[^A-Za-z0-9._:-]+")
_WEB_ACTIONABLE_FOCUS = frozenset(
    {
        "web_query_waves",
        "official_source",
        "independent_evidence_families",
        "time_qualified_observation",
        "visual_or_satellite_evidence",
    }
)
_SOURCE_TYPE_PRIORITY = {
    "official": 0,
    "satellite": 1,
    "press": 2,
    "panoramax": 3,
    "metadata": 4,
    "witness": 5,
    "social": 6,
    "other": 7,
}


def _policy(
    publisher: str,
    source_type: str,
    independence_weight: float,
) -> SourceDomainPolicy:
    return SourceDomainPolicy.model_validate(
        {
            "publisher": publisher,
            "source_type": source_type,
            "independence_weight": independence_weight,
        }
    )


DEFAULT_SOURCE_POLICIES: dict[str, SourceDomainPolicy] = {
    "interieur.gouv.fr": _policy("French Ministry of Interior", "official", 1.0),
    "pompiers.fr": _policy("Sapeurs-pompiers de France", "official", 0.95),
    "georisques.gouv.fr": _policy("Georisques", "official", 1.0),
    "ec.europa.eu": _policy("European Commission", "official", 1.0),
    "copernicus.eu": _policy("Copernicus", "satellite", 1.0),
    "francetvinfo.fr": _policy("Franceinfo", "press", 0.9),
    "ici.fr": _policy("Ici", "press", 0.9),
    "france24.com": _policy("France 24", "press", 0.9),
    "lemonde.fr": _policy("Le Monde", "press", 0.9),
    "lefigaro.fr": _policy("Le Figaro", "press", 0.85),
    "bfmtv.com": _policy("BFMTV", "press", 0.8),
    "tf1info.fr": _policy("TF1 Info", "press", 0.8),
    "20minutes.fr": _policy("20 Minutes", "press", 0.8),
    "ouest-france.fr": _policy("Ouest-France", "press", 0.9),
    "sudouest.fr": _policy("Sud Ouest", "press", 0.9),
    "ladepeche.fr": _policy("La Depeche", "press", 0.85),
    "midilibre.fr": _policy("Midi Libre", "press", 0.85),
    "laprovence.com": _policy("La Provence", "press", 0.85),
    "nice-matin.com": _policy("Nice-Matin", "press", 0.85),
    "varmatin.com": _policy("Var-Matin", "press", 0.85),
    "corsematin.com": _policy("Corse-Matin", "press", 0.85),
    "ledauphine.com": _policy("Le Dauphine Libere", "press", 0.85),
    "reuters.com": _policy("Reuters", "press", 0.95),
    "apnews.com": _policy("Associated Press", "press", 0.95),
    "euronews.com": _policy("Euronews", "press", 0.85),
    "bbc.com": _policy("BBC", "press", 0.9),
    "theguardian.com": _policy("The Guardian", "press", 0.85),
}


class AutomaticSourcePlannerConfig(StrictModel):
    search_provider_domain: str = "html.duckduckgo.com"
    search_template: str = "https://html.duckduckgo.com/html/?q={query}"
    source_policies: dict[str, SourceDomainPolicy] = Field(
        default_factory=lambda: dict(DEFAULT_SOURCE_POLICIES)
    )
    media_ticket_limit: int = Field(default=100, ge=1, le=2_048)
    video_ticket_limit: int = Field(default=30, ge=0, le=512)
    max_source_pages: int = Field(default=200, ge=1, le=10_000)
    results_per_page: int = Field(default=20, ge=1, le=50)
    media_per_source: int = Field(default=8, ge=1, le=20)
    max_pages_per_run: int = Field(default=5, ge=1, le=50)
    max_multimodal_analyses_per_run: int = Field(default=20, ge=1, le=100)
    max_site_qualified_queries: int = Field(default=32, ge=0, le=64)

    @model_validator(mode="after")
    def validate_domains(self) -> AutomaticSourcePlannerConfig:
        normalized = {key.casefold().rstrip(".") for key in self.source_policies}
        if len(normalized) != len(self.source_policies):
            raise ValueError("automatic source domains must be unique")
        if self.search_provider_domain.casefold().rstrip(".") in normalized:
            raise ValueError("search provider must be separate from source domains")
        if self.video_ticket_limit > self.media_ticket_limit:
            raise ValueError("video ticket limit cannot exceed the total media ticket limit")
        return self


def _safe_public_label(value: str | None) -> str | None:
    if value is None or _UNSAFE_LABEL_PATTERN.search(value):
        return None
    normalized = " ".join(_LABEL_CHARACTER_PATTERN.sub(" ", value).split())
    if not 2 <= len(normalized) <= 120:
        return None
    return normalized


def _safe_focus(value: str) -> str | None:
    normalized = _FOCUS_PATTERN.sub("_", value.strip().casefold()).strip("_")
    return normalized[:128] if normalized else None


class AutomaticSourceAcquisitionPlanner:
    """Build a stable plan without accepting user-supplied domains or templates."""

    def __init__(self, config: AutomaticSourcePlannerConfig | None = None) -> None:
        self.config = config or AutomaticSourcePlannerConfig()

    def build(self, durable: DurableEventEvidence) -> SourceAcquisitionPlan:
        observed = durable.event.time_window.from_at
        date = observed.date().isoformat() if observed is not None else ""
        label = _safe_public_label(durable.viewpoint_label)
        progress = durable.research_progress
        wave_focus: tuple[str, ...]
        if progress is None:
            wave_number = 1
            wave_focus = (
                "incident_identity",
                "official_state",
                "daily_progression",
                "visual_evidence",
                "lifecycle_transition",
            )
        elif (
            progress.completed
            and progress.wave_number < 16
            and (
                not progress.converged
                or (
                    durable.incident_day_coverage is not None
                    and not durable.incident_day_coverage.documentary_ready
                )
            )
        ):
            wave_number = progress.wave_number + 1
            coverage = durable.incident_day_coverage
            raw_missing = coverage.missing_dimensions if coverage is not None else ()
            wave_focus = tuple(
                dict.fromkeys(
                    focus
                    for item in raw_missing
                    if (focus := _safe_focus(item)) is not None
                    and (focus in _WEB_ACTIONABLE_FOCUS or focus.startswith("lifecycle:"))
                )
            ) or ("collection_convergence",)
        else:
            wave_number = progress.wave_number
            wave_focus = progress.wave_focus
        subject = f'"{label}"' if label else ""
        query_terms: list[str] = []
        focus_queries = {
            "incident_identity": ("incendie", "feu de foret"),
            "official_state": ("point de situation prefecture sdis",),
            "daily_progression": (
                "progression front secteur",
                "hectares pompiers evacuation",
            ),
            "visual_evidence": ("photo video drone fumee flammes",),
            "lifecycle_transition": ("reprise feu fixe maitrise circonscrit eteint",),
            "official_source": ("communique prefecture sdis mairie onf",),
            "independent_evidence_families": ("reportage temoignage presse locale",),
            "time_qualified_observation": ("heure chronologie point de situation",),
            "visual_or_satellite_evidence": ("photo video drone satellite",),
            "spatial_observation": ("secteur lieu carte drone satellite",),
            "web_query_waves": (
                "incendie",
                "point de situation",
            ),
            "collection_convergence": (
                "bilan chronologie",
                "dernieres informations",
            ),
        }
        for focus in wave_focus:
            normalized_focus = focus.split(":", 1)[0]
            if focus.startswith("lifecycle:"):
                normalized_focus = "lifecycle_transition"
            query_terms.extend(focus_queries.get(normalized_focus, (normalized_focus,)))
        query_terms.extend(("bilan chronologie", "dernieres informations"))
        queries = tuple(
            dict.fromkeys(
                " ".join(part for part in (term, subject, date) if part) for term in query_terms
            )
        )
        configured_policies: dict[str, SourceDomainPolicy]
        if durable.research_source_policies:
            configured_policies = {
                domain: SourceDomainPolicy.model_validate(policy)
                for domain, policy in durable.research_source_policies.items()
            }
        else:
            configured_policies = self.config.source_policies
        policies = {
            domain.casefold().rstrip("."): policy for domain, policy in configured_policies.items()
        }
        ordered_domains = sorted(
            policies,
            key=lambda domain: (
                _SOURCE_TYPE_PRIORITY.get(policies[domain].source_type, 99),
                -policies[domain].independence_weight,
                domain,
            ),
        )
        site_queries = tuple(
            " ".join(part for part in ("incendie", subject, date, f"site:{domain}") if part)
            for domain in ordered_domains[: self.config.max_site_qualified_queries]
        )
        queries = tuple(dict.fromkeys((*queries, *site_queries)))
        search_provider_domain = self.config.search_provider_domain
        search_template = self.config.search_template
        if durable.research_search_templates:
            preferred = search_provider_domain.casefold().rstrip(".")
            if preferred in durable.research_search_templates:
                search_provider_domain = preferred
                search_template = durable.research_search_templates[preferred]
            else:
                search_provider_domain = sorted(durable.research_search_templates)[0]
                search_template = durable.research_search_templates[search_provider_domain]
        identity = json.dumps(
            {
                "candidate_id": durable.event.event_id,
                "wave_number": wave_number,
                "wave_focus": wave_focus,
                "queries": queries,
                "domains": sorted(policies),
                "media_ticket_limit": self.config.media_ticket_limit,
                "video_ticket_limit": self.config.video_ticket_limit,
                "max_source_pages": self.config.max_source_pages,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return SourceAcquisitionPlan(
            candidate_id=durable.event.event_id,
            plan_id=f"PLAN-AUTO-{sha256(identity.encode('utf-8')).hexdigest()[:24]}",
            wave_number=wave_number,
            wave_focus=wave_focus,
            queries=queries,
            allowed_domains=tuple(sorted(policies)),
            source_policies=policies,
            search_provider_domain=search_provider_domain,
            search_template=search_template,
            media_ticket_limit=self.config.media_ticket_limit,
            video_ticket_limit=self.config.video_ticket_limit,
            max_source_pages=self.config.max_source_pages,
            results_per_page=self.config.results_per_page,
            media_per_source=self.config.media_per_source,
            max_pages_per_run=self.config.max_pages_per_run,
            max_multimodal_analyses_per_run=(self.config.max_multimodal_analyses_per_run),
        )


__all__ = [
    "DEFAULT_SOURCE_POLICIES",
    "AutomaticSourceAcquisitionPlanner",
    "AutomaticSourcePlannerConfig",
]
