from __future__ import annotations

import re
from typing import Callable


CONFIRMED = "confirmed_from_ticket"
INFERRED = "inferred_from_research"
GENERATED = "generated_suggestion"

ResearchProvider = Callable[[dict], dict]


def build_research_enrichment(ticket_seed: dict, ocr_text: str, ocr_lines: list[str], provider: ResearchProvider | None = None) -> dict:
    """
    Build a research packet that is safe to generate offline today and easy to
    replace with a real research provider later.
    """
    if provider is not None:
        external = provider(ticket_seed)
        if external:
            return external

    artist = ticket_seed.get("artist", "").strip()
    venue = ticket_seed.get("venue", "").strip()
    city = ticket_seed.get("city", "").strip()
    state = ticket_seed.get("state", "").strip()
    exact_date = ticket_seed.get("exactDate", "").strip()
    year = ticket_seed.get("year", "").strip()

    date_display = exact_date or year or "unknown date"
    location_display = ", ".join(part for part in [city, state] if part) or "unknown location"

    title_line = f"{artist} at {venue}" if artist and venue else artist or venue or "This show"
    venue_date_note = (
        f"OCR suggests {title_line} on {date_display} in {location_display}. "
        "Confirm against the ticket text, personal records, and a trusted outside source before publishing."
    )

    likely_tour_note = (
        f"Look for setlists and tour context around {artist} performances near {date_display}. "
        "Use this to confirm headliner/opener roles and identify likely songs from that run."
    ).strip()

    editorial_suggestion = _build_editorial_suggestion(ticket_seed)
    tag_suggestions = _build_tag_suggestions(ticket_seed)

    return {
        "status": "placeholder_ready_for_live_research",
        "provider": {
            "name": "offline_placeholder",
            "liveLookupUsed": False,
            "description": "No live web research was performed in this environment. Suggestions are structured placeholders for human review."
        },
        "requiresHumanReview": True,
        "researchGaps": [
            "Confirm headliner and opener billing from an authoritative source.",
            "Confirm venue naming and the exact date if OCR quality is weak.",
            "Find one or more likely setlist references for the tour date range.",
            "Verify contextual background before using any editorial copy."
        ],
        "suggestions": {
            "lineup": [
                {
                    "artist": item.get("name", ""),
                    "role": item.get("role", "primary"),
                    "confidenceLabel": CONFIRMED if item.get("role") in {"primary", "headliner"} else GENERATED,
                    "basis": "Human-reviewed ticket metadata" if item.get("role") in {"primary", "headliner"} else "Placeholder suggestion pending research"
                }
                for item in ticket_seed.get("artists", [])
            ],
            "venueDateConfirmation": {
                "note": venue_date_note,
                "confidenceLabel": INFERRED,
                "basis": "OCR-derived metadata plus normalized user-confirmed fields"
            },
            "likelySetlistReferences": [
                {
                    "description": likely_tour_note,
                    "confidenceLabel": GENERATED,
                    "basis": "Template guidance for later research"
                }
            ],
            "contextualShowBackground": [
                {
                    "description": _build_context_suggestion(ticket_seed),
                    "confidenceLabel": GENERATED,
                    "basis": "Generated suggestion from known ticket fields"
                }
            ],
            "editorialCopySuggestions": [
                {
                    "text": editorial_suggestion,
                    "confidenceLabel": GENERATED,
                    "basis": "Generated writing prompt from confirmed metadata"
                }
            ],
            "tagSuggestions": [
                {
                    "tag": tag,
                    "confidenceLabel": GENERATED,
                    "basis": "Generated from ticket metadata and existing archive style"
                }
                for tag in tag_suggestions
            ]
        },
        "ocrSignals": {
            "lineCount": len(ocr_lines),
            "artistMentions": _count_mentions(ocr_text, artist),
            "venueMentions": _count_mentions(ocr_text, venue),
            "dateMentions": _count_mentions(ocr_text, exact_date) if exact_date else 0
        }
    }


def build_provenance_map(ticket: dict, ocr_text: str) -> dict:
    artist = ticket.get("artist", "")
    venue = ticket.get("venue", "")
    exact_date = ticket.get("exactDate", "")
    price = ticket.get("price", "")

    return {
        "artist": _field_entry(artist, CONFIRMED if _count_mentions(ocr_text, artist) else GENERATED, "Display artist reviewed during ingest"),
        "artistSlug": _field_entry(ticket.get("artistSlug", ""), GENERATED, "Derived from display artist"),
        "artists": _field_entry(ticket.get("artists", []), GENERATED, "Structured artist roles derived during ingest"),
        "searchableArtistSlugs": _field_entry(ticket.get("searchableArtistSlugs", []), GENERATED, "Derived search helper fields"),
        "exactDate": _field_entry(exact_date, CONFIRMED if _count_mentions(ocr_text, exact_date) else GENERATED, "OCR suggestion confirmed or entered during ingest"),
        "year": _field_entry(ticket.get("year", ""), CONFIRMED if _year_seen_in_ocr(ocr_text, ticket.get("year", "")) else GENERATED, "OCR suggestion confirmed or entered during ingest"),
        "venue": _field_entry(venue, CONFIRMED if _count_mentions(ocr_text, venue) else GENERATED, "Venue normalized during ingest"),
        "city": _field_entry(ticket.get("city", ""), GENERATED, "Location normalized during ingest"),
        "state": _field_entry(ticket.get("state", ""), GENERATED, "Location normalized during ingest"),
        "country": _field_entry(ticket.get("country", ""), GENERATED, "Defaulted or confirmed during ingest"),
        "copy": _field_entry(ticket.get("copy", ""), GENERATED, "Human-authored or AI-assisted draft copy"),
        "extendedNotes": _field_entry(ticket.get("extendedNotes", ""), GENERATED, "Human-authored extended notes"),
        "companions": _field_entry(ticket.get("companions", []), GENERATED, "Entered manually during ingest"),
        "photos": _field_entry(ticket.get("photos", []), GENERATED, "Entered manually during ingest"),
        "youtubeUrl": _field_entry(ticket.get("youtubeUrl", ""), GENERATED, "Entered manually during ingest"),
        "price": _field_entry(price, CONFIRMED if _count_mentions(ocr_text, price) else GENERATED, "OCR suggestion confirmed or entered during ingest"),
        "tags": _field_entry(ticket.get("tags", []), GENERATED, "Human-entered or AI-suggested tags"),
        "shareTitle": _field_entry(ticket.get("shareTitle", ""), GENERATED, "Derived share metadata"),
        "shareDescription": _field_entry(ticket.get("shareDescription", ""), GENERATED, "Derived share metadata"),
        "shareImage": _field_entry(ticket.get("shareImage", ""), GENERATED, "Derived filename"),
        "slug": _field_entry(ticket.get("slug", ""), GENERATED, "Derived from artist, venue, and year"),
        "img": _field_entry(ticket.get("img", ""), GENERATED, "Derived filename"),
        "rotation": _field_entry(ticket.get("rotation", ""), GENERATED, "Manual presentation setting")
    }


def _field_entry(value, label: str, basis: str) -> dict:
    return {
        "value": value,
        "confidenceLabel": label,
        "basis": basis
    }


def _build_editorial_suggestion(ticket_seed: dict) -> str:
    artist = ticket_seed.get("artist", "This show")
    venue = ticket_seed.get("venue", "the venue")
    city = ticket_seed.get("city", "the city")
    year = ticket_seed.get("year", "that year")
    return (
        f"{artist} at {venue} in {city} ({year}) stands out as a memory worth anchoring in a specific detail: "
        "what drew you to the show, what the room felt like, and what made the performance stick."
    )


def _build_context_suggestion(ticket_seed: dict) -> str:
    artist = ticket_seed.get("artist", "the artist")
    exact_date = ticket_seed.get("exactDate", "") or ticket_seed.get("year", "the period")
    return (
        f"Research the tour cycle around {artist} on or near {exact_date}, including the album era, "
        "notable opener history, and whether this date was part of a special run, festival, or reunion."
    )


def _build_tag_suggestions(ticket_seed: dict) -> list[str]:
    suggestions: list[str] = []
    year = ticket_seed.get("year", "").strip()
    city = ticket_seed.get("city", "").strip().lower()
    venue = ticket_seed.get("venue", "").strip().lower()
    price = ticket_seed.get("price", "").strip()

    if year:
        decade = _decade_tag(year)
        if decade:
            suggestions.append(decade)

    if city:
        suggestions.append(city)

    if venue:
        suggestions.append(venue)

    if price:
        suggestions.append("ticket stub")

    return _dedupe_preserve_order(suggestions)


def _decade_tag(year: str) -> str:
    if not re.fullmatch(r"\d{4}", year):
        return ""
    return f"{year[:3]}0s"


def _year_seen_in_ocr(text: str, year: str) -> bool:
    return bool(year and year in text)


def _count_mentions(text: str, phrase: str) -> int:
    if not phrase:
        return 0
    normalized_text = _normalize(text)
    normalized_phrase = _normalize(phrase)
    if not normalized_phrase:
        return 0
    return normalized_text.count(normalized_phrase)


def _normalize(value: str) -> str:
    value = value.casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique
