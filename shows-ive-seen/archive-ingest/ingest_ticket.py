from pathlib import Path
import shutil
import json
import re
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import pytesseract

# Point directly to your confirmed Tesseract install
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
    "hammerstein", "bowery", "garden", "lunch"
]

EXCLUDE_ARTIST_WORDS = {
    "admission", "section", "row", "seat", "price", "event", "code", "barcode",
    "ticket", "tickets", "adult", "general", "standing", "orch", "mezz", "loge",
    "balcony", "no refunds", "refund", "exchange", "service", "handling",
    "facility", "charge", "includes", "admit", "pm", "am", "doors", "open",
    "website", "com", "ticketmaster", "livenation", "barclays", "center",
    "theatre", "theater", "stadium", "arena", "garden", "hall", "presents",
    "productions", "world", "tour", "subject", "time", "date"
}

MONTHS = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
    "JUL": "07", "AUG": "08", "SEP": "09", "SEPT": "09", "OCT": "10", "NOV": "11", "DEC": "12"
}


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def js_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


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

    # Gentle denoise while preserving ticket structure
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Contrast boost for faded text
    gray = cv2.convertScaleAbs(gray, alpha=1.25, beta=5)

    # Adaptive threshold tends to help old ticket stock
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


def get_lines(text: str):
    lines = [clean_line(line) for line in text.splitlines()]
    return [line for line in lines if line]


def extract_year(text: str) -> str:
    years = re.findall(r"\b(19[6-9]\d|20[0-4]\d)\b", text)
    return years[0] if years else ""


def extract_exact_date(text: str) -> str:
    text_upper = text.upper()

    # Examples: FRI OCT 27 2017 / TUE MAR 21, 2017 / THURSDAY JAN 24 2008
    month_pattern = "|".join(MONTHS.keys())
    m = re.search(rf"\b(?:MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY|MON|TUE|WED|THU|THUR|FRI|SAT|SUN)?\s*({month_pattern})\s+(\d{{1,2}}),?\s+(19[6-9]\d|20[0-4]\d)\b", text_upper)
    if m:
        month = MONTHS[m.group(1)]
        day = m.group(2).zfill(2)
        year = m.group(3)
        return f"{year}-{month}-{day}"

    # Numeric dates like 10/27/2017 or 10-27-2017
    m2 = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](19[6-9]\d|20[0-4]\d)\b", text_upper)
    if m2:
        month = m2.group(1).zfill(2)
        day = m2.group(2).zfill(2)
        year = m2.group(3)
        return f"{year}-{month}-{day}"

    return ""


def extract_price(text: str) -> str:
    prices = re.findall(r"\$\s?\d+(?:\.\d{2})?", text)
    if not prices:
        return ""
    # Prefer the largest non-zero price
    cleaned = []
    for p in prices:
        value = p.replace(" ", "")
        try:
            amt = float(value.replace("$", ""))
            cleaned.append((amt, value))
        except ValueError:
            pass

    if not cleaned:
        return prices[0].replace(" ", "")

    non_zero = [item for item in cleaned if item[0] > 0]
    target = max(non_zero, default=max(cleaned, key=lambda x: x[0]), key=lambda x: x[0])
    return target[1]


def extract_state(text: str) -> str:
    text_upper = text.upper()

    # Exact 2-letter state abbreviations near common city patterns
    for abbr in sorted(set(STATE_MAP.values())):
        if re.search(rf"\b{abbr}\b", text_upper):
            return abbr

    for name, abbr in STATE_MAP.items():
        if re.search(rf"\b{re.escape(name.upper())}\b", text_upper):
            return abbr

    return ""


def extract_city(text: str) -> str:
    text_lower = text.lower()

    for alias, canonical in CITY_ALIASES.items():
        if alias in text_lower:
            return canonical

    # Try patterns like "Austin, TX" or "NYC"
    m = re.search(r"\b(New York|Brooklyn|Austin|Boston|Chicago|Los Angeles|San Francisco|Nashville|Atlanta|Seattle|Philadelphia|Washington)\b", text, flags=re.IGNORECASE)
    if m:
        return normalize_city(m.group(1))

    return ""


def looks_like_venue(line: str) -> bool:
    lowered = line.lower()
    return any(keyword in lowered for keyword in KNOWN_VENUE_KEYWORDS)


def looks_like_artist(line: str) -> bool:
    lowered = line.lower()

    if len(line) < 3:
        return False

    if any(word in lowered for word in EXCLUDE_ARTIST_WORDS):
        return False

    if re.search(r"\b(19[6-9]\d|20[0-4]\d)\b", line):
        return False

    if "$" in line:
        return False

    # Allow caps-heavy lines and artist separator styles
    alpha_chars = sum(c.isalpha() for c in line)
    upper_chars = sum(c.isupper() for c in line)
    if alpha_chars >= 4 and upper_chars / max(alpha_chars, 1) > 0.6:
        return True

    if "&" in line or "feat" in lowered:
        return True

    return False


def extract_venue(lines) -> str:
    venue_candidates = []
    for line in lines:
        if looks_like_venue(line):
            venue_candidates.append(line)

    if venue_candidates:
        # Prefer shorter venue-like line over address-like line
        venue_candidates = sorted(venue_candidates, key=lambda x: (len(x), x))
        return venue_candidates[0]

    return ""


def extract_artist(lines, venue="", city="", year="") -> str:
    candidates = []

    for line in lines:
        lowered = line.lower()

        if venue and venue.lower() in lowered:
            continue
        if city and city.lower() in lowered:
            continue
        if year and year in line:
            continue
        if looks_like_artist(line):
            candidates.append(line)

    if not candidates:
        return ""

    # Prefer lines that are not too long and contain mostly letters
    candidates = sorted(candidates, key=lambda x: (abs(len(x) - 18), -sum(c.isalpha() for c in x)))
    return candidates[0].title() if candidates[0].isupper() else candidates[0]


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


def build_share_page(ticket: dict, site_base_url: str) -> str:
    slug = ticket["slug"]
    share_title = ticket["shareTitle"]
    share_description = ticket["shareDescription"]
    share_image_url = f"{site_base_url}/{ticket['shareImage']}"
    share_url = f"{site_base_url}/share/{slug}/"
    deep_link_url = f"{site_base_url}/#{slug}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{share_title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <meta name="description" content="{share_description}">

  <meta property="og:type" content="website">
  <meta property="og:title" content="{share_title}">
  <meta property="og:description" content="{share_description}">
  <meta property="og:url" content="{share_url}">
  <meta property="og:site_name" content="Anthony C. Dorsey">

  <meta property="og:image" content="{share_image_url}">
  <meta property="og:image:secure_url" content="{share_image_url}">
  <meta property="og:image:type" content="image/jpeg">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="630">
  <meta property="og:image:alt" content="{ticket['artist']} concert ticket from {ticket['venue']} in {ticket['year']}">

  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="{share_title}">
  <meta name="twitter:description" content="{share_description}">
  <meta name="twitter:image" content="{share_image_url}">

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


def main():
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

    print("\nShows I Saw - OCR Ingest Helper v1.1\n")

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

    artist = prompt_with_default("Artist", guessed_artist, required=True)
    venue = prompt_with_default("Venue", guessed_venue, required=True)
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

    artist_slug = slugify(artist)
    venue_slug = slugify(venue)
    slug = f"{artist_slug}-{venue_slug}-{year}"

    final_image_filename = f"{slug}.jpg"
    share_image_filename = f"{slug}-share.jpg"
    json_filename = f"{slug}.json"
    notes_filename = f"{slug}.txt"
    share_page_filename = f"{slug}.html"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    copied_original_name = f"{timestamp}-{source_file.name}"
    copied_original_path = originals_dir / copied_original_name
    shutil.copy2(source_file, copied_original_path)

    companions = [item.strip() for item in companions_raw.split(",") if item.strip()]
    photos = [item.strip() for item in photos_raw.split(",") if item.strip()]
    tags = [item.strip() for item in tags_raw.split(",") if item.strip()]

    share_title = f"{artist} at {venue}, {year} | Shows I Saw"
    share_description = copy_text if copy_text else f"{artist} at {venue} in {city}, {year}."

    ticket = {
        "artist": artist,
        "artistSlug": artist_slug,
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
        "sourceOriginal": str(copied_original_path.relative_to(project_root)).replace("\\", "/"),
        "selectedIncomingFile": source_file.name,
        "ocrTextPreview": ocr_lines[:30],
        "suggestedShareFolder": f"share/{slug}/",
        "suggestedSharePage": f"share/{slug}/index.html"
    }

    draft_json_path = draft_json_dir / json_filename
    with open(draft_json_path, "w", encoding="utf-8") as f:
        json.dump(ticket, f, indent=2, ensure_ascii=False)

    ticket_object = f"""{{
  artist: "{js_escape(ticket['artist'])}",
  artistSlug: "{ticket['artistSlug']}",
  exactDate: "{ticket['exactDate']}",
  year: "{ticket['year']}",
  venue: "{js_escape(ticket['venue'])}",
  city: "{js_escape(ticket['city'])}",
  state: "{ticket['state']}",
  country: "{js_escape(ticket['country'])}",
  copy: "{js_escape(ticket['copy'])}",
  extendedNotes: "{js_escape(ticket['extendedNotes'])}",
  companions: {json.dumps(ticket['companions'], ensure_ascii=False)},
  photos: {json.dumps(ticket['photos'], ensure_ascii=False)},
  youtubeUrl: "{js_escape(ticket['youtubeUrl'])}",
  price: "{js_escape(ticket['price'])}",
  tags: {json.dumps(ticket['tags'], ensure_ascii=False)},
  shareTitle: "{js_escape(ticket['shareTitle'])}",
  shareDescription: "{js_escape(ticket['shareDescription'])}",
  shareImage: "{ticket['shareImage']}",
  slug: "{ticket['slug']}",
  img: "{ticket['img']}",
  rotation: "{js_escape(ticket['rotation'])}"
}},"""

    share_page_html = build_share_page(ticket, site_base_url)
    share_page_path = share_pages_dir / share_page_filename
    with open(share_page_path, "w", encoding="utf-8") as f:
        f.write(share_page_html)

    notes_text = f"""SHOWS I SAW - OCR INGEST OUTPUT

1. SELECTED INCOMING FILE
{source_file.name}

2. ORIGINAL FILE COPIED TO
{str(copied_original_path)}

3. DRAFT JSON FILE
{str(draft_json_path)}

4. SHARE PAGE DRAFT
{str(share_page_path)}

5. FINAL ROOT IMAGE FILENAME
{final_image_filename}

6. FINAL SHARE IMAGE FILENAME
{share_image_filename}

7. SUGGESTED SHARE FOLDER PATH
share/{slug}/

8. SUGGESTED LIVE SHARE PAGE FILE
share/{slug}/index.html

9. OCR TEXT PREVIEW

{chr(10).join(ocr_lines[:40])}

10. PASTE-READY TICKET OBJECT FOR tickets.js

{ticket_object}

11. PUBLISH CHECKLIST

- Review OCR output in draft JSON
- Review location fields carefully
- Create final root ticket image: {final_image_filename}
- Create final share image: {share_image_filename}
- Paste ticket object into tickets.js
- Copy share page draft into share/{slug}/index.html
- Test the hash link locally
- Deploy only after manual review
"""

    notes_path = notes_dir / notes_filename
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(notes_text)

    print("\nDone.\n")
    print(f"Selected file:      {source_file.name}")
    print(f"Copied original:    {copied_original_path}")
    print(f"Draft JSON:         {draft_json_path}")
    print(f"Notes file:         {notes_path}")
    print(f"Share page draft:   {share_page_path}")
    print("\nFinal normalized values:")
    print(f"- artist:           {artist}")
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
    print("\nThis script does NOT modify the live site automatically.\n")


if __name__ == "__main__":
    main()