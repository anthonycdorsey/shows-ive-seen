from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re

from research_enrichment import build_research_enrichment
from site_writer_support import (
    build_share_description,
    build_share_title,
    build_venue_context,
    find_legacy_curated_match,
    is_blank_value,
    load_venue_lookup,
    merge_ticket_with_fallbacks,
    normalize_artist_key,
    normalize_venue_key,
    parse_existing_ticket_blocks,
)


APPROVAL_REQUIRED_FIELDS = [
    "artist",
    "venue",
    "year",
    "city",
    "state",
    "slug",
    "img",
    "shareTitle",
    "shareDescription",
    "shareImage",
]


@dataclass
class EnrichmentPlan:
    draft_path: Path
    payload: dict
    enriched_payload: dict
    status: str
    fields_filled: list[str]
    field_sources: dict[str, str]
    grouped_sources: dict[str, list[str]]
    remaining_missing: list[str]
    ready_for_approval: bool
    legacy_match: dict
    external_match: dict


def load_external_facts(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_writer_context(archive_dir: Path) -> dict:
    project_root = archive_dir.parent
    tickets_js_path = project_root / "tickets.js"
    venue_lookup_path = archive_dir / "venue_lookup.json"
    tickets_js_text = tickets_js_path.read_text(encoding="utf-8")
    existing_blocks = parse_existing_ticket_blocks(tickets_js_text)
    venue_context = build_venue_context(load_venue_lookup(venue_lookup_path), existing_blocks)
    return {
        "projectRoot": project_root,
        "venueContext": venue_context,
    }


def candidate_draft_paths(archive_dir: Path) -> list[Path]:
    draft_dir = archive_dir / "review" / "draft-json"
    candidates: list[Path] = []
    for path in sorted(draft_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metadata_confidence = payload.get("metadataConfidence", {})
        if payload.get("canonicalNamingCommitted") or metadata_confidence.get("canonicalEligible"):
            candidates.append(path)
    return candidates


def prepare_enrichment_plan(
    draft_path: Path,
    *,
    archive_dir: Path,
    writer_context: dict,
    external_facts: dict | None = None,
) -> EnrichmentPlan:
    payload = json.loads(draft_path.read_text(encoding="utf-8"))
    draft_ticket = dict(payload.get("ticket", {}))
    venue_context = writer_context["venueContext"]
    merged_ticket, completion_debug = merge_ticket_with_fallbacks(
        draft_ticket,
        venue_context=venue_context,
        archive_dir=archive_dir,
        current_draft_path=draft_path,
    )

    enriched_ticket = dict(merged_ticket)
    field_sources = build_field_source_map(draft_ticket, merged_ticket, completion_debug)

    ocr_packet = load_ocr_packet(payload, archive_dir, draft_path)
    legacy_match = completion_debug.get("legacyMatch", {})
    legacy_ticket = load_ticket_from_match(legacy_match)
    style_mode = choose_style_mode(legacy_match, legacy_ticket)

    exact_date = coalesce_field(
        enriched_ticket,
        field_sources,
        "exactDate",
        [
            ("approved_metadata", enriched_ticket.get("exactDate", "")),
            ("legacy_curated", legacy_ticket.get("exactDate", "") if legacy_ticket else ""),
            ("ocr_debug", first_date_candidate(payload)),
        ],
    )
    enriched_ticket["exactDate"] = exact_date

    price = coalesce_field(
        enriched_ticket,
        field_sources,
        "price",
        [
            ("approved_metadata", enriched_ticket.get("price", "")),
            ("legacy_curated", legacy_ticket.get("price", "") if legacy_ticket else ""),
            ("ocr_debug", payload.get("proposedMetadata", {}).get("price", "")),
        ],
    )
    enriched_ticket["price"] = normalize_price(price)

    apply_legacy_content_fields(
        enriched_ticket,
        draft_ticket,
        field_sources,
        legacy_ticket,
        legacy_match,
        payload,
    )

    external_match = match_external_facts(enriched_ticket, payload, external_facts or {})
    apply_external_facts(enriched_ticket, field_sources, external_match)

    if is_blank_value(enriched_ticket.get("copy")):
        enriched_ticket["copy"] = draft_copy_with_style(enriched_ticket, style_mode, legacy_ticket)
        field_sources["copy"] = "auto_drafted"

    if is_blank_value(enriched_ticket.get("extendedNotes")):
        enriched_ticket["extendedNotes"] = draft_extended_notes_with_style(enriched_ticket, style_mode, legacy_ticket)
        field_sources["extendedNotes"] = "auto_drafted"

    if is_blank_value(enriched_ticket.get("tags")):
        enriched_ticket["tags"] = build_enriched_tags(enriched_ticket, legacy_ticket, external_match)
        field_sources["tags"] = "auto_drafted"

    if is_blank_value(enriched_ticket.get("shareTitle")):
        enriched_ticket["shareTitle"] = build_share_title(enriched_ticket)
        field_sources["shareTitle"] = "auto_drafted"

    if is_blank_value(enriched_ticket.get("shareDescription")):
        enriched_ticket["shareDescription"] = build_share_description(enriched_ticket, enriched_ticket.get("copy", ""))
        field_sources["shareDescription"] = "auto_drafted"

    if is_blank_value(enriched_ticket.get("shareImage")) and enriched_ticket.get("slug"):
        enriched_ticket["shareImage"] = f"{enriched_ticket['slug']}-share.jpg"
        field_sources["shareImage"] = "approved_metadata"

    if is_blank_value(enriched_ticket.get("img")) and enriched_ticket.get("slug"):
        enriched_ticket["img"] = f"{enriched_ticket['slug']}.jpg"
        field_sources["img"] = "approved_metadata"

    if is_blank_value(enriched_ticket.get("country")):
        enriched_ticket["country"] = "USA"
        field_sources["country"] = "venue_lookup"

    research_packet = build_research_enrichment(
        enriched_ticket,
        ocr_packet["ocrText"],
        ocr_packet["ocrLines"],
    )

    enrichment_hints = build_enrichment_hints(legacy_ticket, external_match, research_packet)
    remaining_missing = [field for field in APPROVAL_REQUIRED_FIELDS if is_blank_value(enriched_ticket.get(field))]
    ready_for_approval = len(remaining_missing) == 0 and image_available(payload, archive_dir, enriched_ticket)
    status = "READY FOR APPROVAL" if ready_for_approval else "NEEDS MORE REVIEW"

    enriched_payload = dict(payload)
    enriched_payload["ticket"] = enriched_ticket
    enriched_payload["reviewStatus"] = "ready_for_approval" if ready_for_approval else payload.get("reviewStatus", "pending_human_review")
    enriched_payload["futureEnrichment"] = {
        "enabled": bool(external_match.get("matched")),
        "notes": research_packet.get("researchGaps", []),
    }
    enriched_payload["enrichment"] = {
        "styleMode": style_mode,
        "fieldsFilledAutomatically": sorted(
            field for field, source in field_sources.items()
            if field_was_blank(draft_ticket.get(field), enriched_ticket.get(field))
        ),
        "fieldSources": field_sources,
        "groupedFieldSources": group_field_sources(field_sources),
        "remainingMissing": remaining_missing + ([] if image_available(payload, archive_dir, enriched_ticket) else ["image"]),
        "readyForApproval": ready_for_approval,
        "legacyMatch": legacy_match,
        "externalFacts": external_match,
        "researchPacket": research_packet,
        "hints": enrichment_hints,
    }

    return EnrichmentPlan(
        draft_path=draft_path,
        payload=payload,
        enriched_payload=enriched_payload,
        status=status,
        fields_filled=enriched_payload["enrichment"]["fieldsFilledAutomatically"],
        field_sources=field_sources,
        grouped_sources=enriched_payload["enrichment"]["groupedFieldSources"],
        remaining_missing=enriched_payload["enrichment"]["remainingMissing"],
        ready_for_approval=ready_for_approval,
        legacy_match=legacy_match,
        external_match=external_match,
    )


def apply_enrichment_plan(plan: EnrichmentPlan) -> None:
    plan.draft_path.write_text(
        json.dumps(plan.enriched_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_field_source_map(original_ticket: dict, merged_ticket: dict, completion_debug: dict) -> dict[str, str]:
    sources: dict[str, str] = {}
    grouped = completion_debug.get("sources", {})
    for field in grouped.get("approvedMetadata", []):
        sources[field] = "approved_metadata"
    for field in grouped.get("venueLookup", []):
        sources[field] = "venue_lookup"
    for field in grouped.get("legacyCurated", []):
        sources[field] = "legacy_curated"
    for field in grouped.get("autoDrafted", []):
        sources[field] = "auto_drafted"
    for field, value in merged_ticket.items():
        if field not in sources and not is_blank_value(value):
            sources[field] = "approved_metadata" if not field_was_blank(original_ticket.get(field), value) else "approved_metadata"
    return sources


def load_ocr_packet(payload: dict, archive_dir: Path, draft_path: Path) -> dict:
    ticket = payload.get("ticket", {})
    slug = ticket.get("slug", "") or draft_path.stem
    ocr_path = archive_dir / "review" / "ocr" / f"{slug}-ocr-debug.json"
    ocr_debug = {}
    if ocr_path.exists():
        try:
            ocr_debug = json.loads(ocr_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            ocr_debug = {}
    ocr_lines = payload.get("source", {}).get("ocrPreview", []) or []
    ocr_text = "\n".join(ocr_lines)
    if ocr_debug.get("rawOcrTextByPass"):
        ocr_text = "\n\n".join(entry.get("rawText", "") for entry in ocr_debug["rawOcrTextByPass"] if entry.get("rawText"))
    return {
        "ocrText": ocr_text,
        "ocrLines": ocr_lines,
        "ocrDebug": ocr_debug,
    }


def load_ticket_from_match(match: dict) -> dict | None:
    source_path = match.get("sourcePath", "")
    if not source_path:
        return None
    path = Path(source_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(payload.get("ticket"), dict):
        return dict(payload["ticket"])
    return payload if isinstance(payload, dict) else None


def choose_style_mode(legacy_match: dict, legacy_ticket: dict | None) -> str:
    if not legacy_match.get("matched") or legacy_match.get("score", 0) < 5.8 or not legacy_ticket:
        return "concise_editorial"
    copy_text = legacy_ticket.get("copy", "")
    if re.search(r"\b(I|my|me|we|our)\b", copy_text):
        return "light_memory_forward"
    return "concise_editorial"


def apply_legacy_content_fields(
    enriched_ticket: dict,
    draft_ticket: dict,
    field_sources: dict[str, str],
    legacy_ticket: dict | None,
    legacy_match: dict,
    payload: dict,
) -> None:
    if not legacy_ticket or not legacy_match.get("matched"):
        return
    strong_match = legacy_match.get("score", 0) >= 6.2
    prior_sources = payload.get("enrichment", {}).get("fieldSources", {})
    for field in (
        "copy",
        "extendedNotes",
        "tags",
        "companions",
        "youtubeUrl",
        "price",
        "exactDate",
        "shareDescription",
    ):
        legacy_value = legacy_ticket.get(field)
        if is_blank_value(legacy_value):
            continue
        current_value = enriched_ticket.get(field)
        if is_blank_value(current_value):
            enriched_ticket[field] = legacy_value
            field_sources[field] = "legacy_curated"
            continue
        if strong_match and prior_sources.get(field) == "auto_drafted":
            enriched_ticket[field] = legacy_value
            field_sources[field] = "legacy_curated"
            continue
        if strong_match and is_weak_machine_draft(draft_ticket.get(field, "")):
            enriched_ticket[field] = legacy_value
            field_sources[field] = "legacy_curated"


def coalesce_field(ticket: dict, field_sources: dict[str, str], field: str, candidates: list[tuple[str, str]]) -> str:
    current_value = ticket.get(field, "")
    if not is_blank_value(current_value):
        return current_value
    for source, value in candidates:
        if not is_blank_value(value):
            field_sources[field] = source
            return value
    return current_value


def normalize_price(value: str) -> str:
    if not value:
        return ""
    if value.startswith("$"):
        return value
    if re.fullmatch(r"\d+(?:\.\d{2})?", value):
        return f"${value}"
    return value


def first_date_candidate(payload: dict) -> str:
    candidates = payload.get("debug", {}).get("dateCandidates", [])
    if candidates:
        return candidates[0].get("value", "")
    return payload.get("proposedMetadata", {}).get("exactDate", "")


def draft_copy_with_style(ticket: dict, style_mode: str, legacy_ticket: dict | None) -> str:
    artist = ticket.get("artist", "Unknown Artist")
    venue = ticket.get("venue", "Unknown venue")
    city = ticket.get("city", "")
    year = ticket.get("year", "Unknown year")
    date_fragment = ticket.get("exactDate", "") or year
    city_fragment = f" in {city}" if city else ""
    if legacy_ticket and legacy_ticket.get("copy") and style_mode == "light_memory_forward":
        return (
            f"{artist} at {venue}{city_fragment}, {date_fragment}. "
            "This entry keeps a concise, memory-forward tone until fuller personal notes are added."
        )
    return f"{artist} at {venue}{city_fragment}, {date_fragment}. A preserved ticket from the archive."


def draft_extended_notes_with_style(ticket: dict, style_mode: str, legacy_ticket: dict | None) -> str:
    artist = ticket.get("artist", "the artist")
    venue = ticket.get("venue", "the venue")
    city = ticket.get("city", "the city")
    year = ticket.get("year", "that year")
    if style_mode == "light_memory_forward":
        return (
            f"This draft leaves room for the specific memory, but it already anchors {artist} at {venue} in {city} during {year}. "
            "Add what drew you there, what the room felt like, or what made the performance linger."
        )
    return (
        f"This draft keeps the tone concise until a fuller memory is added. "
        f"The ticket anchors {artist} at {venue} in {city or 'the venue city'} during {year}. "
        "Update this with personal context, openers, companions, or anything specific that still feels vivid."
    )


def build_enriched_tags(ticket: dict, legacy_ticket: dict | None, external_match: dict) -> list[str]:
    tags: list[str] = []
    year = ticket.get("year", "")
    venue = ticket.get("venue", "").lower()
    city = ticket.get("city", "").lower()
    if year.startswith("19"):
        tags.append("90s")
    elif year.startswith("200"):
        tags.append("2000s")
    elif year.startswith("201"):
        tags.append("2010s")
    if any(word in venue for word in ("club", "lunch", "lounge", "ballroom")):
        tags.append("club show")
    elif any(word in venue for word in ("center", "arena", "stadium", "theater", "theatre")):
        tags.append("theater show")
    if city:
        tags.append(city)
    if legacy_ticket:
        tags.extend(legacy_ticket.get("tags", [])[:2])
    if external_match.get("facts", {}).get("tour"):
        tags.append("tour context")
    return dedupe_strings(tags)[:5]


def is_weak_machine_draft(value) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip()
    if not normalized:
        return False
    return (
        "A preserved ticket from the archive." in normalized
        or "This draft keeps the tone concise until a fuller memory is added." in normalized
        or "This entry keeps a concise, memory-forward tone until fuller personal notes are added." in normalized
    )


def match_external_facts(ticket: dict, payload: dict, external_facts: dict) -> dict:
    if not external_facts:
        return {"matched": False, "reason": "no external facts configured", "facts": {}}
    candidate_keys = [
        ticket.get("slug", ""),
        normalize_artist_key(ticket.get("artist", "")),
        f"{normalize_artist_key(ticket.get('artist', ''))}-{normalize_venue_key(ticket.get('venue', ''))}-{ticket.get('year', '')}",
        Path(payload.get("source", {}).get("inputFile", "")).stem,
    ]
    for key in candidate_keys:
        if not key:
            continue
        facts = external_facts.get(key)
        if isinstance(facts, dict):
            return {"matched": True, "key": key, "facts": facts}
    return {"matched": False, "reason": "no matching external facts entry", "facts": {}}


def apply_external_facts(ticket: dict, field_sources: dict[str, str], external_match: dict) -> None:
    if not external_match.get("matched"):
        return
    facts = external_match.get("facts", {})
    for field in ("exactDate", "city", "state", "country", "price", "youtubeUrl"):
        if is_blank_value(ticket.get(field)) and not is_blank_value(facts.get(field)):
            ticket[field] = facts[field]
            field_sources[field] = "external_facts"


def build_enrichment_hints(legacy_ticket: dict | None, external_match: dict, research_packet: dict) -> dict:
    hints = {
        "openers": [],
        "tour": "",
        "setlistHints": [],
        "youtubeCandidate": "",
    }
    if legacy_ticket:
        hints["youtubeCandidate"] = legacy_ticket.get("youtubeUrl", "")
    facts = external_match.get("facts", {})
    if isinstance(facts.get("openers"), list):
        hints["openers"] = facts["openers"]
    if isinstance(facts.get("tour"), str):
        hints["tour"] = facts["tour"]
    if isinstance(facts.get("setlistHints"), list):
        hints["setlistHints"] = facts["setlistHints"]
    if not hints["setlistHints"]:
        suggestions = research_packet.get("suggestions", {}).get("likelySetlistReferences", [])
        hints["setlistHints"] = [entry.get("description", "") for entry in suggestions if entry.get("description")]
    return hints


def group_field_sources(field_sources: dict[str, str]) -> dict[str, list[str]]:
    grouped = {
        "OCR / filename priors / current": [],
        "venue lookup": [],
        "legacy curated": [],
        "external facts": [],
        "auto drafted": [],
    }
    for field, source in sorted(field_sources.items()):
        if source in {"approved_metadata", "ocr_debug"}:
            grouped["OCR / filename priors / current"].append(field)
        elif source == "venue_lookup":
            grouped["venue lookup"].append(field)
        elif source == "legacy_curated":
            grouped["legacy curated"].append(field)
        elif source == "external_facts":
            grouped["external facts"].append(field)
        elif source == "auto_drafted":
            grouped["auto drafted"].append(field)
    return grouped


def field_was_blank(original_value, enriched_value) -> bool:
    return is_blank_value(original_value) and not is_blank_value(enriched_value)


def image_available(payload: dict, archive_dir: Path, ticket: dict) -> bool:
    saved_original = payload.get("source", {}).get("savedOriginal", "")
    if saved_original and Path(saved_original).exists():
        return True
    slug = ticket.get("slug", "")
    if not slug:
        return False
    canonical = archive_dir / "originals" / "original-copy" / f"{slug}-original.jpg"
    legacy = archive_dir / "originals" / f"{slug}-original.jpg"
    return canonical.exists() or legacy.exists()


def dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(cleaned)
    return ordered
