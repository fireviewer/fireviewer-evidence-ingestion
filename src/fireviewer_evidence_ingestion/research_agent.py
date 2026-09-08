"""One-shot Qwen source-research agent running inside the seccomp sandbox."""

from __future__ import annotations

import gc
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

from fireviewer_contracts.contracts import ResearchOutputV1
from fireviewer_contracts.model_registry import build_registry, resolve_cached_snapshot
from fireviewer_evidence_ingestion.research_rpc import call

MAX_TOOL_STEPS = 24
MAX_TOOL_TEXT_FOR_MODEL = 12_000

TOOLS = """
Tu disposes uniquement de trois outils via le courtier FireWarning :
- search: {"tool":"search","arguments":{
  "domain":"fournisseur de recherche autorisé","query":"requête"}}
- inspect: {"tool":"inspect","arguments":{"url":"https://..."}}
- fetch: {"tool":"fetch","arguments":{"url":"https://...","store":false,
  "candidate_id":"identifiant-court"}}
Pour une image, vidéo ou audio à conserver, rappelle fetch avec store=true.
Réponds à chaque étape par un unique objet JSON, sans Markdown. Quand la recherche est terminée :
{"final":{"status":"succeeded","candidates":[{"url":"https://...","title":"...",
"published_at":"date ISO 8601 avec fuseau ou null","acquired_at":null,
"media_type":"article|image|video|audio|satellite_image|null",
"license_identifier":null,"attribution":"..."}]}}
Une source finale doit obligatoirement avoir été ouverte avec fetch.
N'invente jamais une URL, une date, une licence, une statistique ni un moyen engagé.
Les publications postérieures à cutoff_at ne doivent pas servir à la journée historique.
Recherche aussi le point d'information quotidien de la mairie concernée.
Cherche en priorité les éléments du récapitulatif situationnel : progression et surface,
évacuations et hébergement, effectifs et moyens terrestres, avions ou hélicoptères,
victimes et dégâts, routes et services interrompus, alertes de pollution, appels aux
dons ou au soutien des secours, consignes et ordres à la population. Conserve chaque
chiffre avec sa source, son heure et son statut officiel ou rapporté.
Inspecte aussi les images et vidéos réellement pertinentes présentes sur les pages.
Elles peuvent être conservées pour l'analyse privée via fetch(store=true), même si leur
licence interdit la republication. Ne marque jamais un média comme publiable sans une
licence ou une autorisation explicite et son crédit.
Respecte source_policies : une source ne peut étayer que ses claim_types.
Météo-France décrit la météo ou le danger, jamais la présence d'un incendie actif.
FIRMS, EFFIS et Copernicus décrivent une anomalie ou une observation satellite, jamais
une évacuation, une extinction, une cause ou un incendie confirmé sans autorité A+.
Une source de presse peut fournir une affirmation situationnelle attribuée ou un média
pour le brouillon privé, mais reste une piste à recouper et ne confirme jamais seule
un fait opérationnel. Mentionne explicitement qu'elle est rapportée et non confirmée.
Ne fusionne jamais deux chiffres contradictoires : conserve chaque valeur, sa source
et son horodatage. publication_policy ne vaut jamais validation de publication.
""".strip()


class ToolClient(Protocol):
    def __call__(self, action: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


def _json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("research model did not return a JSON object")


def _canonical_url(value: str) -> str:
    parts = urlsplit(value)
    host = (parts.hostname or "").casefold().rstrip(".")
    if parts.scheme.casefold() != "https" or not host:
        raise ValueError("research candidate URL must use HTTPS")
    authority = host if parts.port in {None, 443} else f"{host}:{parts.port}"
    return urlunsplit(("https", authority, parts.path or "/", parts.query, ""))


def _candidate_media_fields(
    *,
    candidate: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    media_type = candidate.get("media_type")
    if media_type == "article":
        text = evidence.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("article candidate has no fetched text")
        return {"excerpt": text[:100_000]}
    if media_type in {"image", "video", "audio", "satellite_image"}:
        required = ("blob_pathname", "media_sha256", "size_bytes")
        if not all(evidence.get(field) is not None for field in required):
            raise ValueError("media candidate was not stored by the broker")
        return {field: evidence[field] for field in required}
    return {}


def _build_candidates(
    final: dict[str, Any],
    *,
    fetched: dict[str, dict[str, Any]],
    source_policies: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_candidates = final.get("candidates", [])
    if not isinstance(raw_candidates, list):
        raise ValueError("research final candidates must be a list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_candidates[:500]:
        if not isinstance(raw, dict):
            continue
        canonical = _canonical_url(str(raw.get("url", "")))
        if canonical in seen:
            continue
        evidence = fetched.get(canonical)
        if evidence is None:
            raise ValueError("research final candidate was not fetched through the broker")
        seen.add(canonical)
        source_domain = (urlsplit(canonical).hostname or "").casefold()
        policy_domain = next(
            (
                domain
                for domain in sorted(source_policies, key=len, reverse=True)
                if source_domain == domain or source_domain.endswith(f".{domain}")
            ),
            None,
        )
        if policy_domain is None:
            raise ValueError("research candidate has no source policy")
        candidate = {
            "candidate_id": f"candidate-{len(result) + 1:04d}",
            "canonical_url": canonical,
            "source_domain": source_domain,
            "title": raw.get("title"),
            "published_at": raw.get("published_at"),
            "acquired_at": raw.get("acquired_at"),
            "media_type": raw.get("media_type"),
            "license_identifier": raw.get("license_identifier"),
            "attribution": raw.get("attribution"),
            "provenance": {
                "retrieved_via": "network_broker",
                "retrieved_at": evidence.get("retrieved_at"),
                "content_type": evidence.get("content_type"),
                "sha256": evidence.get("sha256"),
                "source_policy_domain": policy_domain,
                "source_policy": source_policies[policy_domain],
            },
        }
        candidate.update(_candidate_media_fields(candidate=raw, evidence=evidence))
        result.append(candidate)
    return result


def _prompt(payload: dict[str, Any]) -> str:
    context = {
        "research_id": payload.get("research_id"),
        "incident_name": payload.get("incident_name"),
        "incident_reference": payload.get("incident_reference"),
        "location_hint": payload.get("location_hint"),
        "cutoff_at": payload.get("cutoff_at"),
        "local_date": dict(payload.get("analysis_window", {})).get("local_date"),
        "allowed_domains": payload.get("allowed_domains"),
        "source_policies": payload.get("source_policies"),
        "search_provider_domains": sorted(dict(payload.get("search_templates", {}))),
    }
    return f"{TOOLS}\n\nContexte immuable :\n{json.dumps(context, ensure_ascii=False)}"


def _model_reply(model: Any, tokenizer: Any, messages: list[dict[str, str]]) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    generated = model.generate(
        **inputs,
        max_new_tokens=1_024,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    return str(tokenizer.decode(new_tokens, skip_special_tokens=True))


def run_research(payload: dict[str, Any], *, tool_client: ToolClient) -> dict[str, Any]:
    attention = os.getenv("FW_ATTENTION_IMPLEMENTATION", "")
    if attention != "flash_attention_2":
        raise RuntimeError("source research requires flash_attention_2")
    import torch
    import transformers

    spec = build_registry()["source_research"]
    snapshot = resolve_cached_snapshot(
        spec,
        Path(os.getenv("FW_HF_CACHE_ROOT", "/runpod-volume/huggingface-cache/hub")),
    )
    started_at = datetime.now(UTC)
    load_started = perf_counter()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = transformers.AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        max_memory={
            0: f"{int(os.getenv('FW_QWEN_GPU_MEMORY_GIB', '44'))}GiB",
            "cpu": f"{int(os.getenv('FW_QWEN_CPU_MEMORY_GIB', '48'))}GiB",
        },
        low_cpu_mem_usage=True,
    )
    model.eval()  # type: ignore[no-untyped-call]
    load_ms = round((perf_counter() - load_started) * 1_000)
    messages = [{"role": "system", "content": _prompt(payload)}]
    queries: list[str] = []
    fetched: dict[str, dict[str, Any]] = {}
    inference_started = perf_counter()
    final: dict[str, Any] | None = None
    try:
        for _step in range(MAX_TOOL_STEPS):
            reply = _model_reply(model, tokenizer, messages)
            decision = _json_object(reply)
            messages.append({"role": "assistant", "content": json.dumps(decision)})
            if "final" in decision:
                value = decision["final"]
                if not isinstance(value, dict):
                    raise ValueError("research final payload must be an object")
                final = value
                break
            action = str(decision.get("tool", ""))
            arguments = decision.get("arguments")
            if action not in {"search", "fetch", "inspect"} or not isinstance(arguments, dict):
                raise ValueError("research model requested an invalid tool call")
            result = tool_client(action, arguments)
            if action == "search":
                query = str(arguments.get("query", "")).strip()
                if query:
                    queries.append(query)
            elif action == "fetch":
                requested = _canonical_url(str(arguments.get("url", "")))
                fetched[requested] = result
                returned_url = result.get("url")
                if isinstance(returned_url, str):
                    fetched[_canonical_url(returned_url)] = result
            model_result = json.dumps(result, ensure_ascii=False)
            messages.append(
                {
                    "role": "user",
                    "content": f"Résultat outil {action}: {model_result[:MAX_TOOL_TEXT_FOR_MODEL]}",
                }
            )
        if final is None:
            raise RuntimeError("research model exhausted the tool-step limit")
        source_policies = payload.get("source_policies")
        if not isinstance(source_policies, dict):
            raise ValueError("research source policies are unavailable")
        candidates = _build_candidates(
            final,
            fetched=fetched,
            source_policies=source_policies,
        )
        finished_at = datetime.now(UTC)
        output = ResearchOutputV1.model_validate(
            {
                "research_id": payload["research_id"],
                "status": final.get("status", "succeeded"),
                "retryable": False,
                "model_run": {
                    "model_id": spec.model_id,
                    "revision": spec.revision,
                    "status": "succeeded",
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "load_ms": load_ms,
                    "inference_ms": round((perf_counter() - inference_started) * 1_000),
                    "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                },
                "queries": queries,
                "candidates": candidates,
                "validation_errors": [],
            }
        )
        return output.model_dump(mode="json")
    finally:
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # type: ignore[no-untyped-call]


def main() -> None:
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise SystemExit("research input must be a JSON object")
    socket_path = os.getenv("FW_RESEARCH_BROKER_SOCKET", "")
    session_token = os.getenv("FW_RESEARCH_SESSION_TOKEN", "")
    if not socket_path or len(session_token) < 32:
        raise SystemExit("research broker session is unavailable")

    def broker_tool(action: str, arguments: dict[str, Any]) -> dict[str, Any]:
        response = call(
            socket_path,
            {"action": action, "session_token": session_token, "arguments": arguments},
        )
        if response.get("ok") is not True or not isinstance(response.get("result"), dict):
            raise RuntimeError(str(response.get("error") or "research broker failed"))
        return dict(response["result"])

    json.dump(run_research(payload, tool_client=broker_tool), sys.stdout, separators=(",", ":"))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
