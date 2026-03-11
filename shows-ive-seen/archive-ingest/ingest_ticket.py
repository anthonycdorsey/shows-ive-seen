from pathlib import Path
import json
import re
import shutil
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import pytesseract

from research_enrichment import build_provenance_map, build_research_enrichment

# Point directly to your Tesseract install.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

STATE_MAP = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
    "dc": "DC", "district of columbia": "DC"
}

CITY_ALIASES = {
    "nyc": "New York",
    "new york city": "New York",
    "new york": "New York",
    "brooklyn": "Brooklyn",
    "austin": "Austin",
    "boston": "Boston",
    "chicago": "Chicago",
    "los angeles": "Los Angeles",
    "san francisco": "San Francisco",
    "nashville": "Nashville",
    "atlanta": "Atlanta",
    "seattle": "Seattle",
    "washington": "Washington",
    "philadelphia": "Philadelphia",
}

KNOWN_VENUE_KEYWORDS = [
    "stadium", "theatre", "theater", "center", "centre", "arena", "hall",
    "garden", "club", "pavilion", "ballroom", "bowl", "lounge", "auditorium",
    "amphitheatre", "amphitheater", "room", "palace", "barclays", "beacon",
    "hammerstein", "bowery", "lunch", "forest hills"
]

ADDRESS_WORDS = [
    "broadway", "street", "st.", "st ", "avenue", "ave", "road", "rd", "blvd",
    "boulevard", "lane", "ln", "drive", "dr", "highway", "hwy", "west", "east",
    "north", "south", "nyc"
]

EXCLUDE_ARTIST_WORDS = {
    "admission", "section", "row", "seat", "price", "event", "code", "barcode",
    "ticket", "tickets", "adult", "general", "standing", "orch", "mezz", "loge",
    "balcony", "refund", "exchange", "service", "handling", "facility", "charge",
    "includes", "admit", "pm", "am", "doors", "open", "website", "com",
    "ticketmaster", "livenation", "barclays", "center", "theatre", "theater",
    "stadium", "arena", "garden", "hall", "presents", "productions", "subject",
    "time", "date", "broadway", "street", "avenue"
}

VENUE_NORMALIZATION = {
    "BEACON THEATRE": "Beacon Theatre",
    "BEACON THEATER": "Beacon Theatre",
    "FOREST HILLS STADIUM": "Forest Hills Stadium",
    "BARCLAYS CENTER": "Barclays Center",
    "HAMMERSTEIN BALLROOM": "Hammerstein Ballroom",
    "BOWERY BALLROOM": "Bowery Ballroom",
    "MADISON SQUARE GARDEN": "Madison Square Garden",
    "FRANK ERWIN CENTER": "Frank Erwin Center",
    "LIBERTY LUNCH": "Liberty Lunch",
    "RADIO CITY MUSIC HALL": "Radio City Music Hall",
}

MONTHS = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
    "JUL": "07", "AUG": "08", "SEP": "09", "SEPT": "09", "OCT": "10", "NOV": "11", "DEC": "12"
}

ROLE_OPTIONS = {"primary", "headliner", "opener", "support", "guest"}


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def js_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def normalize_city(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    lowered = raw.lower().strip()
    lowered = re.sub(r",\s*[A-Z]{2}$", "", lowered)
    if lowered in CITY_ALIASES:
        return CITY_ALIASES[lowered]
    return raw.title() if raw.islower() else raw


def normalize_state(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    lowered = raw.lower()
    if lowered in STATE_MAP:
        return STATE_MAP[lowered]
    if re.fullmatch(r"[A-Za-z]{2}", raw):
        return raw.upper()
    return raw.upper()


def normalize_country(value: str) -> str:
    raw = value.strip()
    if not raw:
        return "USA"
    lowered = raw.lower()
    if lowered in {"us", "u.s.", "usa", "u.s.a.", "united states", "united states of america"}:
        return "USA"
    return raw


def normalize_youtube_url(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    if "youtube.com/embed/" in raw:
        match = re.search(r"/embed/([^?&/]+)", raw)
        if match:
            return f"https://youtu.be/{match.group(1)}"
    return raw


def normalize_venue(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    upper = raw.upper()
    if upper in VENUE_NORMALIZATION:
        return VENUE_NORMALIZATION[upper]
    return raw.title() if raw.isupper() else raw


def choose_incoming_file(incoming_dir: Path) -> Path:
    files = sorted(
        [p for p in incoming_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS],
        key=lambda p: p.name.lower()
    )

    if not files:
        raise FileNotFoundError("No supported image files found in archive-ingest/incoming/")

    print("\nAvailable files in incoming/\n")
    for i, path in enumerate(files, start=1):
        print(f"{i}. {path.name}")

    while True:
        choice = input("\nChoose a file number: ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(files):
                return files[idx - 1]
        print("Please enter a valid number from the list above.")


def preprocess_for_ocr(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.convertScaleAbs(gray, alpha=1.25, beta=5)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 11
    )
    return thresh


def run_ocr(image_path: Path) -> str:
    processed = preprocess_for_ocr(image_path)
    pil_img = Image.fromarray(processed)

    text_1 = pytesseract.image_to_string(pil_img, config="--psm 6")
    text_2 = pytesseract.image_to_string(pil_img, config="--psm 11")

    combined = f"{text_1}\n{text_2}"
    combined = combined.replace("\r", "\n")
    combined = re.sub(r"\n{3,}", "\n\n", combined)
    return combined.strip()


def clean_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"\s+", " ", line)
    return line


def get_lines(text: str) -> list[str]:
    lines = [clean_line(line) for line in text.splitlines()]
    return [line for line in lines if line]


def extract_year(text: str) -> str:
    years = re.findall(r"\b(19[6-9]\d|20[0-4]\d)\b", text)
    return years[0] if years else ""


def extract_exact_date(text: str) -> str:
    text_upper = text.upper()
    month_pattern = "|".join(MONTHS.keys())

    match = re.search(
        rf"\b(?:MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY|MON|TUE|WED|THU|THUR|FRI|SAT|SUN)?\s*({month_pattern})\s+(\d{{1,2}}),?\s+(19[6-9]\d|20[0-4]\d)\b",
        text_upper
    )
    if match:
        month = MONTHS[match.group(1)]
        day = match.group(2).zfill(2)
        year = match.group(3)
        return f"{year}-{month}-{day}"

    numeric_match = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](19[6-9]\d|20[0-4]\d)\b", text_upper)
    if numeric_match:
        month = numeric_match.group(1).zfill(2)
        day = numeric_match.group(2).zfill(2)
        year = numeric_match.group(3)
        return f"{year}-{month}-{day}"

    return ""


def extract_price(text: str) -> str:
    candidates = re.findall(r"\$\s?\d+(?:\.\d{2})?", text)
    numeric_candidates = []

    for candidate in candidates:
        clean = candidate.replace(" ", "")
        try:
            numeric_candidates.append((float(clean.replace("$", "")), clean))
        except ValueError:
            continue

    if not numeric_candidates:
        plain = re.findall(r"\b(\d{2,3}\.\d{2})\b", text)
        if plain:
            try:
                highest = max(float(value) for value in plain)
                return f"${highest:.2f}"
            except ValueError:
                return ""
        return ""

    non_zero = [item for item in numeric_candidates if item[0] > 0]
    best = max(non_zero or numeric_candidates, key=lambda item: item[0])

    if best[0] < 10:
        plain = re.findall(r"\b(\d{2,3}\.\d{2})\b", text)
        if plain:
            try:
                highest = max(float(value) for value in plain)
                if highest > best[0]:
                    return f"${highest:.2f}"
            except ValueError:
                pass

    return best[1] if "." in best[1] else f"${best[0]:.2f}"


def extract_state(text: str) -> str:
    text_upper = text.upper()

    city_state_matches = re.findall(r"\b(?:NEW YORK|BROOKLYN|AUSTIN|BOSTON|CHICAGO|LOS ANGELES|SAN FRANCISCO|NASHVILLE|ATLANTA|SEATTLE|PHILADELPHIA|WASHINGTON)[,\s]+([A-Z]{2})\b", text_upper)
    if city_state_matches:
        return city_state_matches[0]

    for name, abbr in STATE_MAP.items():
        if re.search(rf"\b{re.escape(name.upper())}\b", text_upper):
            return abbr

    for abbr in sorted(set(STATE_MAP.values())):
        if re.search(rf"\b{abbr}\b", text_upper):
            return abbr

    return ""


def extract_city(text: str) -> str:
    text_lower = text.lower()
    for alias, canonical in CITY_ALIASES.items():
        if alias in text_lower:
            return canonical

    match = re.search(r"\b(New York|Brooklyn|Austin|Boston|Chicago|Los Angeles|San Francisco|Nashville|Atlanta|Seattle|Philadelphia|Washington)\b", text, flags=re.IGNORECASE)
    if match:
        return normalize_city(match.group(1))
    return ""


def looks_like_address(line: str) -> bool:
    lowered = line.lower()
    if any(word in lowered for word in ADDRESS_WORDS):
        return True
    if re.search(r"\b\d{2,5}\b", line) and any(word in lowered for word in ADDRESS_WORDS):
        return True
    return False


def looks_like_venue(line: str) -> bool:
    lowered = line.lower()
    return any(keyword in lowered for keyword in KNOWN_VENUE_KEYWORDS)


def looks_like_artist(line: str) -> bool:
    lowered = line.lower()

    if len(line) < 3:
        return False
    if looks_like_address(line):
        return False
    if any(word in lowered for word in EXCLUDE_ARTIST_WORDS):
        return False
    if re.search(r"\b(19[6-9]\d|20[0-4]\d)\b", line):
        return False
    if "$" in line:
        return False

    alpha_chars = sum(char.isalpha() for char in line)
    upper_chars = sum(char.isupper() for char in line)

    if "&" in line:
        return True

    if alpha_chars >= 4 and upper_chars / max(alpha_chars, 1) > 0.6:
        return True

    return False


def extract_venue(lines: list[str]) -> str:
    candidates = []
    for line in lines:
        if looks_like_venue(line) and not looks_like_address(line):
            candidates.append(line)

    if candidates:
        candidates = sorted(candidates, key=lambda value: (len(value), value))
        return normalize_venue(candidates[0])

    return ""


def extract_artist(lines: list[str], venue: str = "", city: str = "", year: str = "") -> str:
    candidates = []

    for line in lines:
        lowered = line.lower()

        if venue and venue.lower() in lowered:
            continue
        if city and city.lower() in lowered:
            continue
        if year and year in line:
            continue
        if looks_like_address(line):
            continue

        if looks_like_artist(line):
            score = 0

            if "&" in line:
                score += 10
            if line.isupper():
                score += 6
            if "PRESENTS" in line.upper():
                score -= 10
            if "THE BOWERY PRESENTS" in line.upper():
                score -= 10
            if len(line) <= 40:
                score += 3
            if any(word in lowered for word in EXCLUDE_ARTIST_WORDS):
                score -= 8

            candidates.append((score, line))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: (-item[0], abs(len(item[1]) - 20)))
    best = candidates[0][1]
    return best.title() if best.isupper() else best


def prompt_with_default(label: str, default: str = "", required: bool = False) -> str:
    while True:
        if default:
            value = input(f"{label} [{default}]: ").strip()
            final = value if value else default
        else:
            final = input(f"{label}: ").strip()

        if required and not final:
            print("This field is required.")
            continue
        return final


def prompt_role(label: str, default: str = "primary") -> str:
    while True:
        value = prompt_with_default(label, default).strip().lower()
        if value in ROLE_OPTIONS:
            return value
        print("Role must be one of: primary, headliner, opener, support, guest")


def parse_collaboration_display_artist(display_artist: str) -> list[str]:
    text = display_artist.strip()

    if " & " in text:
        return [part.strip() for part in text.split(" & ") if part.strip()]

    if " and " in text.lower():
        return [part.strip() for part in re.split(r"\band\b", text, flags=re.IGNORECASE) if part.strip()]

    if " / " in text:
        return [part.strip() for part in text.split(" / ") if part.strip()]

    return [text]


def build_artist_fields(display_artist: str, display_role: str, additional_artists_raw: str):
    display_artist = display_artist.strip()
    display_artist_slug = slugify(display_artist)

    artists = []
    searchable = []

    display_parts = parse_collaboration_display_artist(display_artist)

    if len(display_parts) > 1 and display_role == "primary":
        for part in display_parts:
            artists.append({
                "name": part,
                "slug": slugify(part),
                "role": "primary"
            })
            searchable.append(slugify(part))
        searchable.append(display_artist_slug)
    else:
        artists.append({
            "name": display_artist,
            "slug": display_artist_slug,
            "role": display_role
        })
        searchable.append(display_artist_slug)

    additional_items = [item.strip() for item in additional_artists_raw.split(",") if item.strip()]

    for item in additional_items:
        role = prompt_role(f"Role for {item}", "opener")
        item_slug = slugify(item)
        artists.append({
            "name": item,
            "slug": item_slug,
            "role": role
        })
        searchable.append(item_slug)

    searchable_unique = []
    for item_slug in searchable:
        if item_slug not in searchable_unique:
            searchable_unique.append(item_slug)

    return display_artist_slug, artists, searchable_unique


def build_share_page(ticket: dict, site_base_url: str) -> str:
    slug = ticket["slug"]
    share_title = ticket["shareTitle"]
    share_description = ticket["shareDescription"]
    share_image_url = f"{site_base_url}/{ticket['shareImage']}"
    share_url = f"{site_base_url}/share/{slug}/"
    deep_link_url = f"{site_base_url}/#{slug}"

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\">
  <title>{share_title}</title>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">

  <meta name=\"description\" content=\"{share_description}\">

  <meta property=\"og:type\" content=\"website\">
  <meta property=\"og:title\" content=\"{share_title}\">
  <meta property=\"og:description\" content=\"{share_description}\">
  <meta property=\"og:url\" content=\"{share_url}\">
  <meta property=\"og:site_name\" content=\"Anthony C. Dorsey\">

  <meta property=\"og:image\" content=\"{share_image_url}\">
  <meta property=\"og:image:secure_url\" content=\"{share_image_url}\">
  <meta property=\"og:image:type\" content=\"image/jpeg\">
  <meta property=\"og:image:width\" content=\"1200\">
  <meta property=\"og:image:height\" content=\"630\">
  <meta property=\"og:image:alt\" content=\"{ticket['artist']} concert ticket from {ticket['venue']} in {ticket['year']}\">

  <meta name=\"twitter:card\" content=\"summary_large_image\">
  <meta name=\"twitter:title\" content=\"{share_title}\">
  <meta name=\"twitter:description\" content=\"{share_description}\">
  <meta name=\"twitter:image\" content=\"{share_image_url}\">

  <link rel=\"canonical\" href=\"{share_url}\">

  <script>
    window.location.replace(\"{deep_link_url}\");
  </script>
</head>
<body>
  <p>Opening ticket...</p>
</body>
</html>
"""


def reserve_output_path(path: Path) -> Path:
    if not path.exists():
        return path

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return path.with_name(f"{path.stem}-{timestamp}{path.suffix}")


def build_ticket_object(ticket: dict) -> str:
    ordered_keys = [
        "artist",
        "artistSlug",
        "artists",
        "searchableArtistSlugs",
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
    ]

    lines = ["{"]
    for key in ordered_keys:
        value = ticket.get(key)
        lines.append(f"  {key}: {to_js_literal(value)},")
    lines.append("},")
    return "\n".join(lines)


def to_js_literal(value, indent: int = 2) -> str:
    if isinstance(value, str):
        return f'"{js_escape(value)}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(isinstance(item, str) for item in value):
            return "[" + ", ".join(to_js_literal(item, indent) for item in value) + "]"
        child_indent = indent + 2
        item_lines = ["["]
        for item in value:
            item_literal = to_js_literal(item, child_indent)
            indented = indent_multiline(item_literal, child_indent)
            item_lines.append(f"{' ' * child_indent}{indented},")
        item_lines.append(" " * indent + "]")
        return "\n".join(item_lines)
    if isinstance(value, dict):
        if not value:
            return "{}"
        child_indent = indent + 2
        object_lines = ["{"]
        for key, item in value.items():
            item_literal = to_js_literal(item, child_indent)
            indented = indent_multiline(item_literal, child_indent)
            object_lines.append(f"{' ' * child_indent}{key}: {indented},")
        object_lines.append(" " * indent + "}")
        return "\n".join(object_lines)
    return json.dumps(value, ensure_ascii=False)


def indent_multiline(value: str, indent: int) -> str:
    prefix = " " * indent
    return value.replace("\n", f"\n{prefix}")


def build_notes_text(
    source_file: Path,
    copied_original_path: Path,
    draft_json_path: Path,
    share_page_path: Path,
    final_image_filename: str,
    share_image_filename: str,
    slug: str,
    ocr_lines: list[str],
    ticket_object: str,
    ticket: dict,
    provenance: dict,
    research: dict,
) -> str:
    provenance_text = json.dumps(provenance, indent=2, ensure_ascii=False)
    research_text = json.dumps(research, indent=2, ensure_ascii=False)

    return f"""SHOWS I SAW - INGEST REVIEW PACKAGE

1. SELECTED INCOMING FILE
{source_file.name}

2. ORIGINAL FILE COPIED TO
{copied_original_path}

3. DRAFT JSON FILE
{draft_json_path}

4. SHARE PAGE DRAFT
{share_page_path}

5. FINAL ROOT IMAGE FILENAME
{final_image_filename}

6. FINAL SHARE IMAGE FILENAME
{share_image_filename}

7. SUGGESTED SHARE FOLDER PATH
share/{slug}/

8. SUGGESTED LIVE SHARE PAGE FILE
share/{slug}/index.html

9. NORMALIZED TICKET SUMMARY
Artist: {ticket['artist']}
Artist Slug: {ticket['artistSlug']}
Venue: {ticket['venue']}
City: {ticket['city']}
State: {ticket['state']}
Country: {ticket['country']}
Year: {ticket['year']}
Exact Date: {ticket['exactDate']}
Price: {ticket['price']}
Slug: {ticket['slug']}

10. OCR TEXT PREVIEW

{chr(10).join(ocr_lines[:40])}

11. PROVENANCE / CONFIDENCE MAP

{provenance_text}

12. RESEARCH ENRICHMENT PLACEHOLDERS

{research_text}

13. PASTE-READY TICKET OBJECT FOR tickets.js

{ticket_object}

14. HUMAN REVIEW CHECKLIST

- Review OCR output in the draft JSON.
- Review provenance labels before trusting any value.
- Treat every research suggestion as a draft until you confirm it manually.
- Confirm headliner and opener roles.
- Confirm venue naming and exact date.
- Confirm any setlist references before using them.
- Create the final root ticket image: {final_image_filename}
- Create the final share image: {share_image_filename}
- Paste the approved ticket object into tickets.js manually.
- Copy the approved share page draft into share/{slug}/index.html manually.
- Test the hash link locally.
- Deploy only after manual review.

This tool only creates review artifacts. It never updates the live site automatically.
"""


def main() -> None:
    script_path = Path(__file__).resolve()
    archive_ingest_dir = script_path.parent
    project_root = archive_ingest_dir.parent

    incoming_dir = archive_ingest_dir / "incoming"
    originals_dir = archive_ingest_dir / "originals"
    draft_json_dir = archive_ingest_dir / "review" / "draft-json"
    notes_dir = archive_ingest_dir / "review" / "notes"
    share_pages_dir = archive_ingest_dir / "review" / "share-pages"

    for folder in [incoming_dir, originals_dir, draft_json_dir, notes_dir, share_pages_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    site_base_url = "https://www.anthonycdorsey.com/shows-ive-seen"

    print("\nShows I Saw - OCR Ingest Helper v2.0\n")
    print("This tool creates review artifacts only. It never updates the live site automatically.\n")

    source_file = choose_incoming_file(incoming_dir)
    print(f"\nSelected file: {source_file.name}")
    print("\nRunning OCR... please wait.\n")

    ocr_text = run_ocr(source_file)
    ocr_lines = get_lines(ocr_text)

    guessed_year = extract_year(ocr_text)
    guessed_date = extract_exact_date(ocr_text)
    guessed_price = extract_price(ocr_text)
    guessed_city = extract_city(ocr_text)
    guessed_state = extract_state(ocr_text)
    guessed_venue = extract_venue(ocr_lines)
    guessed_artist = extract_artist(ocr_lines, venue=guessed_venue, city=guessed_city, year=guessed_year)
    guessed_country = "USA"

    print("OCR suggestions:")
    print(f"- artist:     {guessed_artist}")
    print(f"- venue:      {guessed_venue}")
    print(f"- city:       {guessed_city}")
    print(f"- state:      {guessed_state}")
    print(f"- country:    {guessed_country}")
    print(f"- year:       {guessed_year}")
    print(f"- exactDate:  {guessed_date}")
    print(f"- price:      {guessed_price}")
    print()

    display_artist = prompt_with_default("Display artist", guessed_artist, required=True)
    display_role = prompt_role("Role for display artist", "primary")
    additional_artists_raw = prompt_with_default("Additional billed artists/openers (comma-separated)", "")

    artist_slug, artists, searchable_artist_slugs = build_artist_fields(
        display_artist=display_artist,
        display_role=display_role,
        additional_artists_raw=additional_artists_raw,
    )

    venue = normalize_venue(prompt_with_default("Venue", guessed_venue, required=True))
    city = normalize_city(prompt_with_default("City", guessed_city, required=True))
    state = normalize_state(prompt_with_default("State", guessed_state))
    country = normalize_country(prompt_with_default("Country", guessed_country))
    year = prompt_with_default("Year (YYYY)", guessed_year, required=True)
    exact_date = prompt_with_default("Exact date (YYYY-MM-DD)", guessed_date)
    price = prompt_with_default("Ticket price", guessed_price)

    copy_text = prompt_with_default("Short memory/copy text", "")
    extended_notes = prompt_with_default("Extended notes", "")
    companions_raw = prompt_with_default("Companions (comma-separated)", "")
    photos_raw = prompt_with_default("Photos (comma-separated filenames or paths)", "")
    youtube_url = normalize_youtube_url(prompt_with_default("YouTube URL", ""))
    tags_raw = prompt_with_default("Tags (comma-separated)", "")
    rotation = prompt_with_default("Rotation", "0deg")

    venue_slug = slugify(venue)
    slug = f"{artist_slug}-{venue_slug}-{year}"

    final_image_filename = f"{slug}.jpg"
    share_image_filename = f"{slug}-share.jpg"
    json_filename = f"{slug}.json"
    notes_filename = f"{slug}.txt"
    share_page_filename = f"{slug}.html"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    copied_original_name = f"{timestamp}-{source_file.name}"
    copied_original_path = reserve_output_path(originals_dir / copied_original_name)
    shutil.copy2(source_file, copied_original_path)

    companions = [item.strip() for item in companions_raw.split(",") if item.strip()]
    photos = [item.strip() for item in photos_raw.split(",") if item.strip()]
    tags = [item.strip() for item in tags_raw.split(",") if item.strip()]

    share_title = f"{display_artist} at {venue}, {year} | Shows I Saw"
    share_description = copy_text if copy_text else f"{display_artist} at {venue} in {city}, {year}."

    ticket = {
        "artist": display_artist,
        "artistSlug": artist_slug,
        "artists": artists,
        "searchableArtistSlugs": searchable_artist_slugs,
        "exactDate": exact_date,
        "year": year,
        "venue": venue,
        "city": city,
        "state": state,
        "country": country,
        "copy": copy_text,
        "extendedNotes": extended_notes,
        "companions": companions,
        "photos": photos,
        "youtubeUrl": youtube_url,
        "price": price,
        "tags": tags,
        "shareTitle": share_title,
        "shareDescription": share_description,
        "shareImage": share_image_filename,
        "slug": slug,
        "img": final_image_filename,
        "rotation": rotation,
    }

    provenance = build_provenance_map(ticket, ocr_text)
    research = build_research_enrichment(ticket, ocr_text, ocr_lines)

    review_payload = {
        "reviewStatus": "pending_human_review",
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "ticket": ticket,
        "provenance": provenance,
        "research": research,
        "source": {
            "selectedIncomingFile": source_file.name,
            "sourceOriginal": str(copied_original_path.relative_to(project_root)).replace("\\", "/"),
            "ocrTextPreview": ocr_lines[:40],
        },
        "reviewArtifacts": {
            "suggestedShareFolder": f"share/{slug}/",
            "suggestedSharePage": f"share/{slug}/index.html",
            "finalImageFilename": final_image_filename,
            "finalShareImageFilename": share_image_filename,
        },
    }

    ticket_object = build_ticket_object(ticket)
    review_payload["ticketObjectForTicketsJs"] = ticket_object

    draft_json_path = reserve_output_path(draft_json_dir / json_filename)
    with open(draft_json_path, "w", encoding="utf-8") as file_handle:
        json.dump(review_payload, file_handle, indent=2, ensure_ascii=False)

    share_page_html = build_share_page(ticket, site_base_url)
    share_page_path = reserve_output_path(share_pages_dir / share_page_filename)
    with open(share_page_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(share_page_html)

    notes_text = build_notes_text(
        source_file=source_file,
        copied_original_path=copied_original_path,
        draft_json_path=draft_json_path,
        share_page_path=share_page_path,
        final_image_filename=final_image_filename,
        share_image_filename=share_image_filename,
        slug=slug,
        ocr_lines=ocr_lines,
        ticket_object=ticket_object,
        ticket=ticket,
        provenance=provenance,
        research=research,
    )

    notes_path = reserve_output_path(notes_dir / notes_filename)
    with open(notes_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(notes_text)

    print("\nDone.\n")
    print(f"Selected file:      {source_file.name}")
    print(f"Copied original:    {copied_original_path}")
    print(f"Draft JSON:         {draft_json_path}")
    print(f"Notes file:         {notes_path}")
    print(f"Share page draft:   {share_page_path}")
    print("\nFinal normalized values:")
    print(f"- artist:           {display_artist}")
    print(f"- artistSlug:       {artist_slug}")
    print(f"- artists:          {artists}")
    print(f"- searchable:       {searchable_artist_slugs}")
    print(f"- venue:            {venue}")
    print(f"- city:             {city}")
    print(f"- state:            {state}")
    print(f"- country:          {country}")
    print(f"- year:             {year}")
    print(f"- exactDate:        {exact_date}")
    print(f"- price:            {price}")
    print(f"- slug:             {slug}")
    print(f"- img:              {final_image_filename}")
    print(f"- shareImage:       {share_image_filename}")
    print("\nResearch enrichment provider:")
    print(f"- name:             {research['provider']['name']}")
    print(f"- liveLookupUsed:   {research['provider']['liveLookupUsed']}")
    print("\nThis script does NOT modify the live site automatically.\n")


if __name__ == "__main__":
    main()
