from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import re


SITE_BASE_URL = "https://www.anthonycdorsey.com/shows-ive-seen"
LEGACY_FILL_FIELDS = [
    "exactDate",
    "price",
    "copy",
    "extendedNotes",
    "companions",
    "photos",
    "youtubeUrl",
    "tags",
    "shareTitle",
    "shareDescription",
]
TICKET_FIELDS = [
    "artist",
    "artistSlug",
    "exactDate",
    "year",
    "venue",
    "city",
    "state",
    "country",
    "copy",
    "extendedNotes",
    "companions",
    "photos",
    "youtubeUrl",
    "price",
    "tags",
    "shareTitle",
    "shareDescription",
    "shareImage",
    "slug",
    "img",
    "rotation",
    "notes",
    "youtube",
]


@dataclass
class ExistingTicketBlock:
    slug: str
    start: int
    end: int
    text: str
    fields: dict


def slugify(value: str) -> str:
    text = value.lower().strip()
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def choose_approved_draft(draft_dir: Path) -> Path:
    candidates = []
    for path in sorted(draft_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("canonicalNamingCommitted"):
            candidates.append((path, payload))

    if not candidates:
        raise FileNotFoundError("No approved draft JSON files were found in review/draft-json/")

    print("\nApproved draft files\n")
    for index, (path, payload) in enumerate(candidates, start=1):
        ticket = payload.get("ticket", {})
        print(f"{index}. {path.name} - {ticket.get('artist', '')} / {ticket.get('venue', '')} / {ticket.get('year', '')}")

    while True:
        choice = input("\nChoose a file number: ").strip()
        if choice.isdigit():
            selected = int(choice)
            if 1 <= selected <= len(candidates):
                return candidates[selected - 1][0]
        print("Please enter a valid number.")


def load_venue_lookup(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_venue_key(value: str) -> str:
    text = slugify(value)
    return text.replace("theatre", "theater")


def normalize_artist_key(value: str) -> str:
    return slugify(value)


def parse_existing_ticket_blocks(tickets_js_text: str) -> list[ExistingTicketBlock]:
    blocks: list[ExistingTicketBlock] = []
    array_start = tickets_js_text.find("[")
    array_end = tickets_js_text.rfind("]")
    if array_start == -1 or array_end == -1:
        return blocks

    depth = 0
    object_start = None
    for index in range(array_start, array_end):
        char = tickets_js_text[index]
        if char == "{":
            if depth == 0:
                object_start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and object_start is not None:
                object_end = index + 1
                text = tickets_js_text[object_start:object_end]
                fields = extract_ticket_fields(text)
                slug = fields.get("slug", "")
                blocks.append(ExistingTicketBlock(slug=slug, start=object_start, end=object_end, text=text, fields=fields))
                object_start = None
    return blocks


def extract_ticket_fields(block_text: str) -> dict:
    fields: dict = {}
    for field in TICKET_FIELDS:
        string_match = re.search(rf"{field}:\s*\"((?:[^\"\\]|\\.)*)\"", block_text)
        if string_match:
            fields[field] = string_match.group(1).encode("utf-8").decode("unicode_escape")
            continue
        array_match = re.search(rf"{field}:\s*(\[[^\]]*\])", block_text, flags=re.DOTALL)
        if array_match:
            try:
                fields[field] = json.loads(array_match.group(1))
            except json.JSONDecodeError:
                fields[field] = []
    return fields


def build_venue_context(venue_lookup: dict, existing_blocks: list[ExistingTicketBlock]) -> dict[str, dict]:
    context: dict[str, dict] = {}
    for venue, location in venue_lookup.items():
        context[normalize_venue_key(venue)] = location
    for block in existing_blocks:
        venue = block.fields.get("venue", "")
        if not venue:
            continue
        context.setdefault(
            normalize_venue_key(venue),
            {
                "city": block.fields.get("city", ""),
                "state": block.fields.get("state", ""),
                "country": block.fields.get("country", "USA"),
            },
        )
    return context


def resolve_location(ticket: dict, venue_context: dict[str, dict]) -> tuple[str, str, str]:
    venue = ticket.get("venue", "")
    lookup = venue_context.get(normalize_venue_key(venue), {})
    city = ticket.get("city") or lookup.get("city", "")
    state = ticket.get("state") or lookup.get("state", "")
    country = ticket.get("country") or lookup.get("country", "USA")
    return city, state, country


def resolve_location_with_source(ticket: dict, venue_context: dict[str, dict]) -> tuple[str, str, str, str]:
    venue = ticket.get("venue", "")
    lookup = venue_context.get(normalize_venue_key(venue), {})
    city = ticket.get("city", "")
    state = ticket.get("state", "")
    country = ticket.get("country", "")
    source = "approved_metadata"
    if not city and lookup.get("city"):
        city = lookup.get("city", "")
        source = "venue_lookup"
    if not state and lookup.get("state"):
        state = lookup.get("state", "")
        source = "venue_lookup"
    if not country:
        country = lookup.get("country", "USA")
        if source != "venue_lookup":
            source = "venue_lookup"
    return city, state, country or "USA", source


def draft_copy(ticket: dict, city: str) -> str:
    artist = ticket.get("artist", "Unknown Artist")
    venue = ticket.get("venue", "Unknown venue")
    year = ticket.get("year", "Unknown year")
    city_fragment = f" in {city}" if city else ""
    return f"{artist} at {venue}{city_fragment}, {year}. A preserved ticket from the archive."


def draft_extended_notes(ticket: dict, city: str) -> str:
    artist = ticket.get("artist", "Unknown Artist")
    venue = ticket.get("venue", "Unknown venue")
    year = ticket.get("year", "Unknown year")
    city_fragment = city if city else "the venue"
    return (
        f"This draft keeps the tone concise until a fuller memory is added. "
        f"The ticket anchors {artist} at {venue} in {city_fragment} during {year}. "
        f"Update this with personal context, openers, companions, or anything specific that still feels vivid."
    )


def draft_tags(ticket: dict, city: str) -> list[str]:
    tags: list[str] = []
    year = ticket.get("year", "")
    venue = ticket.get("venue", "").lower()

    if year.startswith("19"):
        tags.append("90s")
    elif year.startswith("20"):
        tags.append("2000s" if year.startswith("200") else "2010s")

    if any(word in venue for word in ("club", "lunch", "lounge", "ballroom")):
        tags.append("club show")
    elif any(word in venue for word in ("center", "arena", "stadium")):
        tags.append("arena show")

    if city:
        tags.append(city.lower())

    unique_tags: list[str] = []
    for tag in tags:
        if tag and tag not in unique_tags:
            unique_tags.append(tag)
    return unique_tags[:4]


def build_share_title(ticket: dict) -> str:
    return f"{ticket['artist']} at {ticket['venue']}, {ticket['year']} | Shows I Saw"


def build_share_description(ticket: dict, copy_text: str) -> str:
    return copy_text


def is_blank_value(value: object) -> bool:
    return value in ("", None) or value == []


def extract_ticket_candidate_from_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("ticket"), dict):
        ticket = dict(payload["ticket"])
        ticket["_sourcePath"] = str(path)
        ticket["_sourceKind"] = "approved_draft"
        ticket["_sourceInputFile"] = payload.get("source", {}).get("inputFile", "")
        return ticket
    if isinstance(payload, dict) and payload.get("artist") and payload.get("venue") and payload.get("year"):
        ticket = dict(payload)
        ticket["_sourcePath"] = str(path)
        ticket["_sourceKind"] = "legacy_json"
        ticket["_sourceInputFile"] = payload.get("selectedIncomingFile", "")
        return ticket
    return None


def token_overlap_score(left: str, right: str) -> float:
    left_tokens = {token for token in normalize_artist_key(left).split("-") if token}
    right_tokens = {token for token in normalize_artist_key(right).split("-") if token}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)


def artist_part_overlap_score(left: str, right: str) -> float:
    left_parts = normalized_artist_parts(left)
    right_parts = normalized_artist_parts(right)
    if not left_parts or not right_parts:
        return 0.0
    matches = 0.0
    for left_part in left_parts:
        best = 0.0
        for right_part in right_parts:
            best = max(best, SequenceMatcher(None, left_part, right_part).ratio())
        matches += best
    return matches / max(len(left_parts), len(right_parts), 1)


def normalized_artist_parts(value: str) -> list[str]:
    normalized = normalize_artist_key(value)
    if not normalized:
        return []
    text = normalized.replace("-and-", "|").replace("-with-", "|")
    text = text.replace("-", " ")
    parts = re.split(r"\s+\|\s+|\|", text)
    cleaned: list[str] = []
    for part in parts:
        tokens = [token for token in part.split() if token]
        if tokens:
            cleaned.append(" ".join(tokens))
    return cleaned


def score_legacy_candidate(current_ticket: dict, candidate_ticket: dict, current_draft_path: Path) -> float:
    score = 0.0
    current_slug = current_ticket.get("slug", "")
    candidate_slug = candidate_ticket.get("slug", "")
    current_venue = current_ticket.get("venue", "")
    candidate_venue = candidate_ticket.get("venue", "")
    current_artist = current_ticket.get("artist", "")
    candidate_artist = candidate_ticket.get("artist", "")
    current_year = current_ticket.get("year", "")
    candidate_year = candidate_ticket.get("year", "")
    current_city = current_ticket.get("city", "")
    candidate_city = candidate_ticket.get("city", "")
    current_state = current_ticket.get("state", "")
    candidate_state = candidate_ticket.get("state", "")

    if normalize_venue_key(current_venue) == normalize_venue_key(candidate_venue):
        score += 3.0
    else:
        score += SequenceMatcher(None, normalize_venue_key(current_venue), normalize_venue_key(candidate_venue)).ratio() * 1.2
    if current_year and current_year == candidate_year:
        score += 2.0

    score += token_overlap_score(current_artist, candidate_artist) * 2.4
    score += artist_part_overlap_score(current_artist, candidate_artist) * 2.6
    if current_slug and candidate_slug:
        score += SequenceMatcher(None, current_slug, candidate_slug).ratio() * 1.2

    input_stem = Path(current_draft_path.name).stem
    candidate_input = candidate_ticket.get("_sourceInputFile", "") or candidate_slug
    if input_stem and candidate_input:
        score += SequenceMatcher(None, slugify(input_stem), slugify(Path(candidate_input).stem)).ratio() * 0.9

    if current_city and candidate_city and slugify(current_city) == slugify(candidate_city):
        score += 0.7
    if current_state and candidate_state and slugify(current_state) == slugify(candidate_state):
        score += 0.6

    if current_artist and candidate_artist:
        current_artist_key = normalize_artist_key(current_artist)
        candidate_artist_key = normalize_artist_key(candidate_artist)
        if current_artist_key in candidate_artist_key or candidate_artist_key in current_artist_key:
            score += 1.5

    return score


def find_legacy_curated_match(current_ticket: dict, archive_dir: Path, current_draft_path: Path) -> tuple[dict | None, dict]:
    draft_dir = archive_dir / "review" / "draft-json"
    candidates: list[tuple[float, dict]] = []
    for path in sorted(draft_dir.glob("*.json")):
        if path.resolve() == current_draft_path.resolve():
            continue
        candidate_ticket = extract_ticket_candidate_from_json(path)
        if not candidate_ticket:
            continue
        score = score_legacy_candidate(current_ticket, candidate_ticket, current_draft_path)
        if score >= 3.2:
            candidates.append((score, candidate_ticket))

    if not candidates:
        return None, {
            "matched": False,
            "reason": "no local legacy curated JSON candidate passed the similarity threshold",
            "topCandidates": [],
        }

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_candidate = candidates[0]
    top_candidates = [
        {
            "score": round(score, 3),
            "sourcePath": candidate.get("_sourcePath", ""),
            "sourceKind": candidate.get("_sourceKind", ""),
            "slug": candidate.get("slug", ""),
            "artist": candidate.get("artist", ""),
            "venue": candidate.get("venue", ""),
            "year": candidate.get("year", ""),
            "importableFields": importable_legacy_fields(current_ticket, candidate),
        }
        for score, candidate in candidates[:3]
    ]
    if best_score < 5.0:
        return None, {
            "matched": False,
            "reason": "legacy candidates were found, but none scored high enough for safe merge",
            "topCandidates": top_candidates,
        }
    return best_candidate, {
        "matched": True,
        "score": round(best_score, 3),
        "sourcePath": best_candidate.get("_sourcePath", ""),
        "sourceKind": best_candidate.get("_sourceKind", ""),
        "slug": best_candidate.get("slug", ""),
        "topCandidates": top_candidates,
        "importableFields": importable_legacy_fields(current_ticket, best_candidate),
    }


def importable_legacy_fields(current_ticket: dict, candidate_ticket: dict) -> list[str]:
    importable: list[str] = []
    for field in LEGACY_FILL_FIELDS:
        current_value = current_ticket.get(field)
        candidate_value = candidate_ticket.get(field)
        if is_blank_value(current_value) and not is_blank_value(candidate_value):
            importable.append(field)
    return importable


def merge_ticket_with_fallbacks(
    draft_ticket: dict,
    *,
    venue_context: dict[str, dict],
    archive_dir: Path,
    current_draft_path: Path,
) -> tuple[dict, dict]:
    merged_ticket = dict(draft_ticket)
    sources = {
        "approvedMetadata": [],
        "venueLookup": [],
        "legacyCurated": [],
        "autoDrafted": [],
        "missing": [],
    }
    for field in ("artist", "venue", "year", "slug", "artistSlug", "img", "rotation"):
        if merged_ticket.get(field):
            sources["approvedMetadata"].append(field)

    city, state, country, location_source = resolve_location_with_source(merged_ticket, venue_context)
    merged_ticket["city"] = city
    merged_ticket["state"] = state
    merged_ticket["country"] = country
    if location_source == "venue_lookup":
        for field in ("city", "state", "country"):
            if merged_ticket.get(field):
                sources["venueLookup"].append(field)
    else:
        for field in ("city", "state", "country"):
            if merged_ticket.get(field):
                sources["approvedMetadata"].append(field)

    legacy_match, legacy_debug = find_legacy_curated_match(merged_ticket, archive_dir, current_draft_path)
    if legacy_match:
        for field in LEGACY_FILL_FIELDS:
            current_value = merged_ticket.get(field)
            legacy_value = legacy_match.get(field)
            is_current_empty = is_blank_value(current_value)
            if is_current_empty and not is_blank_value(legacy_value):
                merged_ticket[field] = legacy_value
                sources["legacyCurated"].append(field)

    copy_text = merged_ticket.get("copy") or draft_copy(merged_ticket, merged_ticket.get("city", ""))
    if not merged_ticket.get("copy"):
        sources["autoDrafted"].append("copy")
    merged_ticket["copy"] = copy_text

    extended_notes = merged_ticket.get("extendedNotes") or draft_extended_notes(merged_ticket, merged_ticket.get("city", ""))
    if not merged_ticket.get("extendedNotes"):
        sources["autoDrafted"].append("extendedNotes")
    merged_ticket["extendedNotes"] = extended_notes

    tags = merged_ticket.get("tags") or draft_tags(merged_ticket, merged_ticket.get("city", ""))
    if not merged_ticket.get("tags"):
        sources["autoDrafted"].append("tags")
    merged_ticket["tags"] = tags

    if not merged_ticket.get("shareTitle"):
        merged_ticket["shareTitle"] = build_share_title(merged_ticket)
        sources["autoDrafted"].append("shareTitle")
    if not merged_ticket.get("shareDescription"):
        merged_ticket["shareDescription"] = build_share_description(merged_ticket, merged_ticket["copy"])
        sources["autoDrafted"].append("shareDescription")

    merged_ticket["companions"] = merged_ticket.get("companions") or []
    merged_ticket["photos"] = merged_ticket.get("photos") or []
    merged_ticket["youtubeUrl"] = merged_ticket.get("youtubeUrl") or ""
    merged_ticket["notes"] = ""
    merged_ticket["youtube"] = ""

    for field in ("artist", "venue", "year", "exactDate", "city", "state", "price", "copy", "extendedNotes", "tags", "shareTitle", "shareDescription"):
        if is_blank_value(merged_ticket.get(field)):
            sources["missing"].append(field)

    return merged_ticket, {
        "sources": sources,
        "legacyMatch": legacy_debug,
    }


def build_site_ticket(draft_payload: dict, venue_context: dict[str, dict], archive_dir: Path, current_draft_path: Path) -> tuple[dict, dict]:
    draft_ticket = dict(draft_payload.get("ticket", {}))
    merged_ticket, completion_debug = merge_ticket_with_fallbacks(
        draft_ticket,
        venue_context=venue_context,
        archive_dir=archive_dir,
        current_draft_path=current_draft_path,
    )
    required_slug = merged_ticket.get("slug", "")
    artist = merged_ticket.get("artist", "")
    venue = merged_ticket.get("venue", "")
    year = merged_ticket.get("year", "")

    return {
        "artist": artist,
        "artistSlug": slugify(artist),
        "exactDate": merged_ticket.get("exactDate", ""),
        "year": year,
        "venue": venue,
        "city": merged_ticket.get("city", ""),
        "state": merged_ticket.get("state", ""),
        "country": merged_ticket.get("country") or "USA",
        "copy": merged_ticket.get("copy", ""),
        "extendedNotes": merged_ticket.get("extendedNotes", ""),
        "companions": merged_ticket.get("companions") or [],
        "photos": merged_ticket.get("photos") or [],
        "youtubeUrl": merged_ticket.get("youtubeUrl") or "",
        "price": merged_ticket.get("price", ""),
        "tags": merged_ticket.get("tags") or [],
        "shareTitle": merged_ticket.get("shareTitle") or build_share_title(merged_ticket),
        "shareDescription": merged_ticket.get("shareDescription") or build_share_description(merged_ticket, merged_ticket.get("copy", "")),
        "shareImage": f"{SITE_BASE_URL}/{required_slug}-share.jpg",
        "slug": required_slug,
        "img": f"{required_slug}.jpg",
        "rotation": merged_ticket.get("rotation") or "0deg",
        "notes": "",
        "youtube": "",
    }, completion_debug


def validate_site_ticket(ticket: dict) -> list[str]:
    required_fields = ["artist", "artistSlug", "year", "venue", "city", "state", "country", "copy", "extendedNotes", "shareTitle", "shareDescription", "shareImage", "slug", "img", "rotation"]
    missing = [field for field in required_fields if not ticket.get(field)]
    return missing


def js_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def format_js_array(values: list[str], indent: str = "        ") -> str:
    if not values:
        return "[]"
    escaped = ", ".join(f"\"{js_escape(value)}\"" for value in values)
    return f"[{escaped}]"


def build_ticket_object_js(ticket: dict) -> str:
    return f"""    {{
        artist: "{js_escape(ticket['artist'])}",
        artistSlug: "{js_escape(ticket['artistSlug'])}",
        exactDate: "{js_escape(ticket['exactDate'])}",
        year: "{js_escape(ticket['year'])}",
        venue: "{js_escape(ticket['venue'])}",
        city: "{js_escape(ticket['city'])}",
        state: "{js_escape(ticket['state'])}",
        country: "{js_escape(ticket['country'])}",
        copy: "{js_escape(ticket['copy'])}",
        extendedNotes: "{js_escape(ticket['extendedNotes'])}",
        companions: {format_js_array(ticket['companions'])},
        photos: {format_js_array(ticket['photos'])},
        youtubeUrl: "{js_escape(ticket['youtubeUrl'])}",
        price: "{js_escape(ticket['price'])}",
        tags: {format_js_array(ticket['tags'])},
        shareTitle: "{js_escape(ticket['shareTitle'])}",
        shareDescription: "{js_escape(ticket['shareDescription'])}",
        shareImage: "{js_escape(ticket['shareImage'])}",
        slug: "{js_escape(ticket['slug'])}",

        img: "{js_escape(ticket['img'])}",
        rotation: "{js_escape(ticket['rotation'])}",

        // temporary backward compatibility
        notes: "{js_escape(ticket['notes'])}",
        youtube: "{js_escape(ticket['youtube'])}"
    }}"""


def render_share_page(ticket: dict) -> str:
    slug = ticket["slug"]
    share_url = f"{SITE_BASE_URL}/share/{slug}/"
    deep_link_url = f"{SITE_BASE_URL}/#{slug}"
    share_image = ticket["shareImage"]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{ticket['shareTitle']}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <meta name="description" content="{ticket['shareDescription']}">

  <meta property="og:type" content="website">
  <meta property="og:title" content="{ticket['shareTitle']}">
  <meta property="og:description" content="{ticket['shareDescription']}">
  <meta property="og:url" content="{share_url}">
  <meta property="og:site_name" content="Anthony C. Dorsey">

  <meta property="og:image" content="{share_image}">
  <meta property="og:image:secure_url" content="{share_image}">
  <meta property="og:image:type" content="image/jpeg">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="630">
  <meta property="og:image:alt" content="{ticket['artist']} concert ticket from {ticket['venue']} in {ticket['year']}">

  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="{ticket['shareTitle']}">
  <meta name="twitter:description" content="{ticket['shareDescription']}">
  <meta name="twitter:image" content="{share_image}">

  <link rel="canonical" href="{share_url}">

  <script>
    window.location.replace("{deep_link_url}");
  </script>
</head>
<body>
  <p>Opening ticket...</p>
</body>
</html>
"""


def build_tickets_js_preview(existing_ticket: dict | None, new_ticket: dict) -> tuple[str, list[str]]:
    changed_fields: list[str] = []
    if existing_ticket:
        for field in TICKET_FIELDS:
            if existing_ticket.get(field) != new_ticket.get(field):
                changed_fields.append(field)
    return build_ticket_object_js(new_ticket), changed_fields


def update_tickets_js(tickets_js_text: str, slug: str, new_object_js: str) -> tuple[str, str]:
    blocks = parse_existing_ticket_blocks(tickets_js_text)
    existing = next((block for block in blocks if block.slug == slug), None)
    if existing:
        updated = tickets_js_text[:existing.start] + new_object_js + tickets_js_text[existing.end:]
        return updated, "UPDATE"

    insert_at = tickets_js_text.rfind("];")
    if insert_at == -1:
        raise ValueError("Could not locate the end of the tickets array in tickets.js")

    prefix = tickets_js_text[:insert_at].rstrip()
    if prefix.endswith("["):
        updated = tickets_js_text[:insert_at] + "\n" + new_object_js + "\n" + tickets_js_text[insert_at:]
    else:
        updated = tickets_js_text[:insert_at].rstrip() + ",\n" + new_object_js + "\n" + tickets_js_text[insert_at:]
    return updated, "ADD"
