from __future__ import annotations

from fireviewer_contracts.mvp.contracts import EventEvidenceV1
from fireviewer_evidence_ingestion.mvp.research.source_planner import (
    AutomaticSourceAcquisitionPlanner,
    AutomaticSourcePlannerConfig,
)
from fireviewer_contracts.mvp.supervision.backend_event_evidence import (
    BackendIncidentDayCoverage,
    DurableEventEvidence,
    DurableResearchProgress,
)


def _durable(*, label: str | None) -> DurableEventEvidence:
    event = EventEvidenceV1.model_validate(
        {
            "event_id": "EC-REAL-1",
            "time_window": {"from_at": "2026-08-23T12:00:00Z"},
        }
    )
    return DurableEventEvidence(
        event=event,
        media_locations=(),
        vision_artifacts=(),
        upload_locations=(),
        prior_fire_states=(),
        geospatial_checks=(),
        geographic_references=(),
        source_revision_sha256="a" * 64,
        viewpoint_label=label,
    )


def test_automatic_planner_builds_stable_queries_without_a_media_completeness_target() -> None:
    planner = AutomaticSourceAcquisitionPlanner()

    first = planner.build(_durable(label="Massif des Maures"))
    replay = planner.build(_durable(label="Massif des Maures"))

    assert first == replay
    assert first.plan_id.startswith("PLAN-AUTO-")
    assert first.media_ticket_limit == 100
    assert first.video_ticket_limit == 30
    assert first.max_source_pages == 200
    assert len(first.source_policies) >= 20
    assert len(first.queries) == 9 + len(first.source_policies)
    assert all("Massif des Maures" in query for query in first.queries)
    assert any("photo video drone" in query for query in first.queries)
    assert any("progression front secteur" in query for query in first.queries)
    assert any("site:interieur.gouv.fr" in query for query in first.queries)
    assert any("site:ledauphine.com" in query for query in first.queries)
    assert first.search_provider_domain not in first.allowed_domains


def test_automatic_planner_does_not_send_contact_data_or_coordinates_to_search() -> None:
    planner = AutomaticSourceAcquisitionPlanner()

    plan = planner.build(_durable(label="06 12 34 56 78 contact@example.org"))

    assert all("contact" not in query for query in plan.queries)
    assert all("example.org" not in query for query in plan.queries)
    assert all("06 12" not in query for query in plan.queries)
    assert plan.queries[0] == "incendie 2026-08-23"


def test_automatic_planner_uses_backend_incident_registry() -> None:
    durable = _durable(label="Die Justin")
    object.__setattr__(
        durable,
        "research_source_policies",
        {
            "drome.gouv.fr": {
                "publisher": "Prefecture de la Drome",
                "source_type": "official",
                "independence_weight": 1.0,
                "claim_types": ["incident_status", "fire_progression"],
            },
            "ledauphine.com": {
                "publisher": "Le Dauphine Libere",
                "source_type": "press",
                "independence_weight": 0.85,
                "claim_types": ["location_report", "area_burned"],
            },
        },
    )
    object.__setattr__(
        durable,
        "research_search_templates",
        {"html.duckduckgo.com": "https://html.duckduckgo.com/html/?q={query}"},
    )

    plan = AutomaticSourceAcquisitionPlanner().build(durable)

    assert set(plan.allowed_domains) == {"drome.gouv.fr", "ledauphine.com"}
    assert plan.source_policies["drome.gouv.fr"].claim_types == (
        "incident_status",
        "fire_progression",
    )
    assert plan.queries[0] == 'incendie "Die Justin" 2026-08-23'
    assert plan.queries[-2:] == (
        'incendie "Die Justin" 2026-08-23 site:drome.gouv.fr',
        'incendie "Die Justin" 2026-08-23 site:ledauphine.com',
    )


def test_automatic_planner_bounds_site_qualified_queries() -> None:
    planner = AutomaticSourceAcquisitionPlanner(
        config=AutomaticSourcePlannerConfig(max_site_qualified_queries=1)
    )

    plan = planner.build(_durable(label="Die Justin"))

    site_queries = tuple(query for query in plan.queries if "site:" in query)
    assert len(site_queries) == 1
    assert site_queries[0].endswith("site:ec.europa.eu")


def test_automatic_planner_opens_a_new_focused_wave_after_partial_completion() -> None:
    durable = _durable(label="Die Justin")
    object.__setattr__(
        durable,
        "research_progress",
        DurableResearchProgress(
            plan_id="PLAN-AUTO-FIRST",
            plan_revision="1" * 64,
            wave_number=1,
            wave_focus=("incident_identity",),
            page_count=9,
            completed=True,
            media_ticket_limit=2_048,
            safety_limit_reached=False,
            converged=False,
            zero_yield_wave_streak=0,
            coverage_ready=False,
            next_cursor=None,
        ),
    )

    plan = AutomaticSourceAcquisitionPlanner().build(durable)

    assert plan.wave_number == 2
    assert plan.wave_focus == ("collection_convergence",)
    assert any("bilan chronologie" in query for query in plan.queries)


def test_converged_wave_opens_a_documentary_gap_wave_but_not_for_satellite_only() -> None:
    progress = DurableResearchProgress(
        plan_id="PLAN-AUTO-FIRST",
        plan_revision="1" * 64,
        wave_number=1,
        wave_focus=("incident_identity",),
        page_count=9,
        completed=True,
        media_ticket_limit=2_048,
        safety_limit_reached=False,
        converged=True,
        zero_yield_wave_streak=2,
        coverage_ready=True,
        next_cursor=None,
    )
    missing_documentary = _durable(label="Die Justin")
    object.__setattr__(missing_documentary, "research_progress", progress)
    object.__setattr__(
        missing_documentary,
        "incident_day_coverage",
        BackendIncidentDayCoverage(
            queries_exhausted=True,
            safety_limit_reached=False,
            converged=True,
            source_count=4,
            official_source_count=0,
            independent_evidence_family_count=2,
            claim_count=3,
            image_count=1,
            video_count=0,
            audio_count=0,
            media_analysis_required_count=0,
            media_analysis_completed_count=0,
            media_analysis_failed_count=0,
            satellite_artifact_count=0,
            materialized_satellite_count=0,
            satellite_analysis_required_count=0,
            satellite_analysis_completed_count=0,
            spatial_observation_count=0,
            time_qualified_observation_count=2,
            expected_lifecycle_phases=("daily_progression_or_status",),
            covered_lifecycle_phases=("daily_progression_or_status",),
            missing_dimensions=("official_source", "spatial_observation"),
            documentary_ready=False,
            spatial_ready=False,
            satellite_analysis_ready=True,
            media_analysis_ready=True,
            coverage_ready=False,
        ),
    )

    follow_up = AutomaticSourceAcquisitionPlanner().build(missing_documentary)

    assert follow_up.wave_number == 2
    assert follow_up.wave_focus == ("official_source",)

    satellite_only = _durable(label="Die Justin")
    object.__setattr__(satellite_only, "research_progress", progress)
    object.__setattr__(
        satellite_only,
        "incident_day_coverage",
        BackendIncidentDayCoverage(
            **{
                **missing_documentary.incident_day_coverage.model_dump(),
                "official_source_count": 1,
                "independent_evidence_family_count": 3,
                "missing_dimensions": ("spatial_observation",),
                "documentary_ready": True,
            }
        ),
    )

    stable = AutomaticSourceAcquisitionPlanner().build(satellite_only)

    assert stable.wave_number == 1
    assert stable.wave_focus == progress.wave_focus
