from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from time import perf_counter
import string

from PIL import Image
import pytesseract


DEFAULT_TESSERACT_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

MONTHS = {
    "JAN": "01",
    "FEB": "02",
    "MAR": "03",
    "APR": "04",
    "MAY": "05",
    "JUN": "06",
    "JUL": "07",
    "AUG": "08",
    "SEP": "09",
    "SEPT": "09",
    "OCT": "10",
    "NOV": "11",
    "DEC": "12",
}

CITY_STATE_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s*,\s*([A-Z]{2})\b"
)

VENUE_KEYWORDS = (
    "amphitheater",
    "amphitheatre",
    "arena",
    "auditorium",
    "ballroom",
    "bowl",
    "coliseum",
    "center",
    "centre",
    "club",
    "garden",
    "hall",
    "lunch",
    "lounge",
    "music hall",
    "opera house",
    "pavilion",
    "stadium",
    "theater",
    "theatre",
)

ARTIST_STOP_WORDS = {
    "admission",
    "adult",
    "barcode",
    "date",
    "doors",
    "event",
    "general",
    "not valid",
    "price",
    "row",
    "seat",
    "section",
    "service charge",
    "tax",
    "ticket",
    "time",
    "venue",
}

SMART_TITLE_EXCEPTIONS = {
    "and": "and",
    "at": "at",
    "da": "da",
    "de": "de",
    "del": "del",
    "for": "for",
    "in": "in",
    "la": "la",
    "of": "of",
    "or": "or",
    "the": "the",
    "to": "to",
    "van": "van",
    "von": "von",
}

NEGATIVE_ARTIST_TERMS = {
    "surcharge",
    "convenience charge",
    "admission",
    "event code",
    "section",
    "row",
    "seat",
    "price",
    "tax",
    "adult",
    "gen adm",
    "general admission",
    "comp",
    "doors",
    "no refund",
    "exchange",
    "re-entry",
    "handling charges",
    "service charges",
    "service charge",
    "handling charge",
    "facility charge",
}


@dataclass
class OCRResult:
    text: str
    lines: list[str]
    average_confidence: float
    best_config: str
    best_variant_label: str
    cleaned_text: str
    debug_runs: list[dict]
    line_entries: list[dict]
    pass_count: int
    early_exit_triggered: bool
    early_exit_reason: str


@dataclass
class MetadataProposal:
    artist: str
    venue: str
    year: str
    exact_date: str
    city: str
    state: str
    country: str
    price: str
    artist_confidence: float
    venue_confidence: float
    year_confidence: float
    date_confidence: float
    core_metadata_confidence: float
    overall_confidence: float
    warnings: list[str]
    artist_candidates: list[dict]
    venue_candidates: list[dict]
    date_candidates: list[dict]
    rejected_artist_candidates: list[dict]
    filename_priors_used: bool
    filename_priors: dict
    approval_blockers: list[str]
    selector_debug: dict

    @property
    def artist_slug(self) -> str:
        return slugify(self.artist) or "unknown-artist"

    @property
    def venue_slug(self) -> str:
        return slugify(self.venue) or "unknown-venue"

    @property
    def canonical_year(self) -> str:
        return self.year or "unknown-year"

    @property
    def slug(self) -> str:
        if self.low_confidence:
            return ""
        return f"{self.artist_slug}-{self.venue_slug}-{self.canonical_year}"

    @property
    def low_confidence(self) -> bool:
        return not self.canonical_eligible

    @property
    def canonical_eligible(self) -> bool:
        return len(self.approval_blockers) == 0


def configure_tesseract(tesseract_cmd: str | None = None) -> Path:
    candidate = Path(tesseract_cmd) if tesseract_cmd else DEFAULT_TESSERACT_PATH
    if not candidate.exists():
        raise FileNotFoundError(
            "Tesseract OCR was not found. Install Tesseract or pass --tesseract-cmd."
        )
    pytesseract.pytesseract.tesseract_cmd = str(candidate)
    return candidate


def run_ocr_variants(images: list[tuple[str, Image.Image]], source_path: Path | None = None) -> OCRResult:
    best_text = ""
    best_lines: list[str] = []
    best_average_confidence = 0.0
    best_score = -1.0
    best_config = "--psm 6"
    best_variant_label = ""
    debug_runs: list[dict] = []
    best_line_entries: list[dict] = []
    weak_run_count = 0
    run_plan = [
        ("--psm 6", "display"),
        ("--psm 6", "ocr-binary"),
        ("--psm 11", "ocr-binary"),
    ]
    image_map = {label: image for label, image in images}
    previous_cleaned_text = ""
    early_exit_triggered = False
    early_exit_reason = ""

    for config, label in run_plan:
        image = image_map.get(label)
        if image is None:
            continue
        run_start = perf_counter()
        timeout_seconds = 4 if label == "display" else 3
        try:
            data = pytesseract.image_to_data(
                image,
                config=config,
                output_type=pytesseract.Output.DICT,
                timeout=timeout_seconds,
            )
        except RuntimeError:
            debug_runs.append(
                {
                    "variant": label,
                    "config": config,
                    "averageConfidence": 0.0,
                    "lineCount": 0,
                    "characterCount": 0,
                    "score": 0.0,
                    "coreMetadataConfidence": 0.0,
                    "runtimeMs": round((perf_counter() - run_start) * 1000, 1),
                    "rawText": "",
                    "cleanedText": "",
                    "lineEntries": [],
                    "timeoutSeconds": timeout_seconds,
                    "timedOut": True,
                }
            )
            weak_run_count += 1
            if weak_run_count >= 1:
                debug_runs[-1]["earlyExit"] = "timeout-stop"
                early_exit_triggered = True
                early_exit_reason = "OCR timed out on a weak ticket pass"
                break
            continue
        words = []
        confidences: list[float] = []
        line_map: dict[tuple[int, int, int], dict] = {}

        for index, (text, confidence) in enumerate(zip(data["text"], data["conf"])):
            cleaned = normalize_space(text)
            if not cleaned:
                continue
            try:
                conf_value = float(confidence)
            except ValueError:
                conf_value = -1.0
            words.append(cleaned)
            if conf_value >= 0:
                confidences.append(conf_value)
            key = (
                int(data["block_num"][index]),
                int(data["par_num"][index]),
                int(data["line_num"][index]),
            )
            entry = line_map.setdefault(
                key,
                {
                    "words": [],
                    "confidences": [],
                    "left": int(data["left"][index]),
                    "top": int(data["top"][index]),
                    "right": int(data["left"][index]) + int(data["width"][index]),
                    "bottom": int(data["top"][index]) + int(data["height"][index]),
                },
            )
            entry["words"].append(cleaned)
            if conf_value >= 0:
                entry["confidences"].append(conf_value)
            entry["left"] = min(entry["left"], int(data["left"][index]))
            entry["top"] = min(entry["top"], int(data["top"][index]))
            entry["right"] = max(entry["right"], int(data["left"][index]) + int(data["width"][index]))
            entry["bottom"] = max(entry["bottom"], int(data["top"][index]) + int(data["height"][index]))

        line_entries = build_line_entries(line_map)
        text = "\n".join(entry["text"] for entry in line_entries)
        lines = [entry["text"] for entry in line_entries]
        average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        score = average_confidence + min(len(" ".join(words)) / 120.0, 1.2)
        quick_artist_candidates, _ = extract_artist_candidates(lines, line_entries)
        quick_venue_candidates = extract_venue_candidates(lines, line_entries)
        quick_year, quick_year_confidence = extract_year(text)
        quick_artist_confidence = float(quick_artist_candidates[0]["score"]) if quick_artist_candidates else 0.0
        quick_venue_confidence = float(quick_venue_candidates[0]["score"]) if quick_venue_candidates else 0.0
        quick_core_confidence = compute_core_metadata_confidence(
            quick_artist_confidence,
            quick_venue_confidence,
            quick_year_confidence,
        )
        runtime_ms = round((perf_counter() - run_start) * 1000, 1)

        debug_runs.append(
            {
                "variant": label,
                "config": config,
                "averageConfidence": round(max(average_confidence / 100.0, 0.0), 3),
                "lineCount": len(lines),
                "characterCount": len(text.strip()),
                "score": round(score, 3),
                "coreMetadataConfidence": round(quick_core_confidence, 3),
                "runtimeMs": runtime_ms,
                "timeoutSeconds": timeout_seconds,
                "rawText": text.strip(),
                "cleanedText": "\n".join(lines),
                "lineEntries": line_entries,
            }
        )

        if score > best_score:
            best_text = text.strip()
            best_lines = lines
            best_average_confidence = average_confidence
            best_score = score
            best_config = f"{config} [{label}]"
            best_variant_label = label
            best_line_entries = line_entries

        current_cleaned_text = "\n".join(lines)
        if quick_core_confidence >= 0.78 and (
            quick_artist_confidence >= 0.60 and quick_venue_confidence >= 0.58 and quick_year_confidence >= 0.70
        ):
            debug_runs[-1]["earlyExit"] = "strong-core-fields"
            early_exit_triggered = True
            early_exit_reason = "core metadata was already strong enough after this OCR pass"
            break

        if quick_core_confidence < 0.24 and average_confidence / 100.0 < 0.22:
            weak_run_count += 1
        else:
            weak_run_count = 0

        if previous_cleaned_text and current_cleaned_text == previous_cleaned_text:
            debug_runs[-1]["earlyExit"] = "duplicate-result"
            early_exit_triggered = True
            early_exit_reason = "a later OCR pass produced the same cleaned text as the previous pass"
            break

        previous_cleaned_text = current_cleaned_text

        if len(debug_runs) >= 2 and quick_core_confidence >= 0.66:
            debug_runs[-1]["earlyExit"] = "good-enough-second-pass"
            early_exit_triggered = True
            early_exit_reason = "two useful OCR passes were enough to identify the core fields"
            break

        if weak_run_count >= 1:
            debug_runs[-1]["earlyExit"] = "weak-pass-stop"
            early_exit_triggered = True
            early_exit_reason = "the first useful OCR pass already indicated extremely weak text"
            break

    return OCRResult(
        text=best_text,
        lines=best_lines,
        average_confidence=max(best_average_confidence / 100.0, 0.0),
        best_config=best_config,
        best_variant_label=best_variant_label,
        cleaned_text="\n".join(best_lines),
        debug_runs=debug_runs,
        line_entries=best_line_entries,
        pass_count=len(debug_runs),
        early_exit_triggered=early_exit_triggered,
        early_exit_reason=early_exit_reason,
    )


def propose_metadata(ocr: OCRResult, source_path: Path | None = None) -> MetadataProposal:
    text = ocr.text
    lines = ocr.lines
    filename_priors = build_filename_priors(source_path) if source_path else empty_filename_priors()
    parser_stages = [
        "ocr_text_selected",
        "artist_candidates_extracted",
        "venue_candidates_extracted",
        "date_candidates_extracted",
        "filename_priors_loaded",
        "multi_artist_parser_checked",
        "venue_tail_split_checked",
        "final_selector_resolved",
    ]

    artist_candidates, rejected_artist_candidates = extract_artist_candidates(lines, ocr.line_entries)
    venue_candidates = extract_venue_candidates(lines, ocr.line_entries)
    date_candidates = extract_date_candidates(text)
    artist, artist_confidence = top_candidate(artist_candidates)
    venue, venue_confidence = top_candidate(venue_candidates)
    year, year_confidence = extract_year(text)
    exact_date, date_confidence = extract_exact_date(text, date_candidates)
    city, state = extract_city_state(text, lines)
    price = extract_price(text)

    filename_priors_used = False
    multi_artist_attempted = True
    multi_artist_matched = False
    venue_tail_split_attempted = bool(venue)
    venue_tail_split_matched = False
    final_selector_winner = "ocr_candidates"
    final_selector_reason = "Top OCR artist/venue/year candidates were used as the starting selection."
    top_artist_word_count = len(artist.split()) if artist else 0
    if filename_priors["artist"] and (
        artist_confidence < 0.56
        or (
            artist
            and top_artist_word_count == 1
            and artist_confidence < 0.72
            and ocr.average_confidence < 0.68
            and looks_like_short_band_name(filename_priors["artist"])
        )
    ):
        artist = filename_priors["artist"]
        artist_confidence = max(artist_confidence, 0.54)
        filename_priors_used = True
        final_selector_winner = "filename_prior_artist"
        final_selector_reason = "Filename prior replaced a weak OCR artist selection."
    multi_artist_from_lines, multi_artist_confidence = detect_multi_artist_billing(lines)
    if multi_artist_from_lines and multi_artist_confidence > artist_confidence:
        artist = multi_artist_from_lines
        artist_confidence = multi_artist_confidence
        multi_artist_matched = True
        final_selector_winner = "multi_artist_line_match"
        final_selector_reason = "Adjacent OCR name lines were combined into a co-billed artist."
    if venue_confidence < 0.5 and filename_priors["venue"]:
        venue = filename_priors["venue"]
        venue_confidence = max(venue_confidence, 0.54)
        filename_priors_used = True
        final_selector_winner = "filename_prior_venue"
        final_selector_reason = "Filename prior replaced a weak OCR venue selection."
    if year_confidence < 0.5 and filename_priors["year"]:
        year = filename_priors["year"]
        year_confidence = max(year_confidence, 0.44)
        filename_priors_used = True

    split_venue = split_artist_prefix_from_venue(venue) if venue else venue
    if venue and split_venue != venue:
        venue = normalize_venue(split_venue)
        venue_confidence = max(venue_confidence, 0.72)
        venue_tail_split_matched = True
        final_selector_winner = "venue_tail_split"
        final_selector_reason = "A mixed OCR venue candidate was split to isolate the venue tail."

    structured_filename_fallback_used = False
    if (
        filename_priors.get("structuredPlausible")
        and year == filename_priors.get("year")
        and (
            artist_confidence < 0.60
            or venue_confidence < 0.72
            or not looks_like_name_shape(artist)
            or has_embedded_artist_tokens_in_venue(venue, filename_priors.get("artistParts", []))
        )
    ):
        artist = filename_priors["artist"]
        venue = filename_priors["venue"]
        year = filename_priors["year"]
        artist_confidence = max(artist_confidence, 0.82)
        venue_confidence = max(venue_confidence, 0.84)
        year_confidence = max(year_confidence, 0.74)
        filename_priors_used = True
        structured_filename_fallback_used = True
        multi_artist_matched = len(filename_priors.get("artistParts", [])) == 2
        final_selector_winner = "structured_filename_fallback"
        final_selector_reason = "Weak/fragmented OCR triggered a structured filename fallback for multi-artist billing."

    if (
        filename_priors.get("artistParts")
        and len(filename_priors["artistParts"]) == 2
        and venue == filename_priors.get("venue")
        and year == filename_priors.get("year")
        and (artist_confidence < 0.66 or not looks_like_name_shape(artist))
    ):
        artist = filename_priors["artist"]
        artist_confidence = max(artist_confidence, 0.82)
        filename_priors_used = True
        multi_artist_matched = True
        final_selector_winner = "multi_artist_filename_rescue"
        final_selector_reason = "Venue-tail splitting and structured filename priors rescued a weak multi-artist OCR result."

    if venue and looks_like_short_clean_venue(venue):
        venue_confidence = max(venue_confidence, 0.62)

    warnings: list[str] = []
    if not artist:
        warnings.append("Could not confidently identify the artist from OCR text.")
    if not venue:
        warnings.append("Could not confidently identify the venue from OCR text.")
    if not year:
        warnings.append("Could not confidently identify the year from OCR text.")
    if ocr.average_confidence < 0.55:
        warnings.append("Overall OCR confidence is weak; review text carefully.")
    if exact_date and year and not exact_date.startswith(year):
        warnings.append("Parsed exact date does not align with the strongest year candidate.")

    core_metadata_confidence = compute_core_metadata_confidence(
        artist_confidence,
        venue_confidence,
        year_confidence,
    )
    overall_confidence = (
        (artist_confidence * 0.30)
        + (venue_confidence * 0.34)
        + (year_confidence * 0.20)
        + (min(ocr.average_confidence, 1.0) * 0.16)
    )
    if artist_confidence < 0.58:
        overall_confidence = min(overall_confidence, 0.56)
    if artist_confidence < 0.42:
        overall_confidence = min(overall_confidence, 0.42)
    approval_blockers = build_approval_blockers(
        artist=artist,
        venue=venue,
        year=year,
        artist_confidence=artist_confidence,
        venue_confidence=venue_confidence,
        year_confidence=year_confidence,
        core_metadata_confidence=core_metadata_confidence,
    )

    return MetadataProposal(
        artist=artist,
        venue=venue,
        year=year,
        exact_date=exact_date,
        city=city,
        state=state,
        country="USA",
        price=price,
        artist_confidence=artist_confidence,
        venue_confidence=venue_confidence,
        year_confidence=year_confidence,
        date_confidence=date_confidence,
        core_metadata_confidence=core_metadata_confidence,
        overall_confidence=min(overall_confidence, 0.99),
        warnings=warnings,
        artist_candidates=artist_candidates[:8],
        venue_candidates=venue_candidates[:8],
        date_candidates=date_candidates[:8],
        rejected_artist_candidates=rejected_artist_candidates[:12],
        filename_priors_used=filename_priors_used,
        filename_priors={**filename_priors, "used": filename_priors_used},
        approval_blockers=approval_blockers,
        selector_debug={
            "parserStagesRun": parser_stages,
            "multiArtistParsingAttempted": multi_artist_attempted,
            "multiArtistParsingMatched": multi_artist_matched,
            "venueTailSplittingAttempted": venue_tail_split_attempted,
            "venueTailSplittingMatched": venue_tail_split_matched,
            "structuredFilenameFallbackUsed": structured_filename_fallback_used,
            "finalSelectorWinner": final_selector_winner,
            "finalSelectorReason": final_selector_reason,
            "finalArtist": artist,
            "finalVenue": venue,
            "finalYear": year,
            "functionChain": [
                "run_ocr_variants",
                "extract_artist_candidates",
                "extract_venue_candidates",
                "extract_year",
                "extract_exact_date",
                "build_filename_priors",
                "detect_multi_artist_billing",
                "split_artist_prefix_from_venue",
                "final_selection_resolution",
            ],
        },
    )


def slugify(value: str) -> str:
    text = value.lower().strip()
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def build_review_slug(source_path: Path) -> str:
    stem = slugify(source_path.stem) or "ticket"
    return f"review-{stem}"


def build_draft_ticket(proposal: MetadataProposal) -> dict:
    slug = proposal.slug if not proposal.low_confidence else ""
    year_text = proposal.year or ""
    venue_text = proposal.venue or ""
    artist_text = proposal.artist or ""

    return {
        "artist": artist_text,
        "artistSlug": proposal.artist_slug,
        "exactDate": proposal.exact_date,
        "year": year_text,
        "venue": venue_text,
        "city": proposal.city,
        "state": proposal.state,
        "country": proposal.country,
        "copy": "",
        "extendedNotes": "",
        "companions": [],
        "photos": [],
        "youtubeUrl": "",
        "price": proposal.price,
        "tags": [],
        "shareTitle": f"{artist_text or 'Unknown Artist'} at {venue_text or 'Unknown Venue'}, {year_text or 'Unknown Year'} | Shows I Saw",
        "shareDescription": "",
        "shareImage": f"{slug}-share.jpg" if slug else "",
        "slug": slug,
        "img": f"{slug}.jpg" if slug else "",
        "rotation": "0deg",
    }


def build_review_payload(
    source_path: Path,
    draft_ticket: dict,
    proposal: MetadataProposal,
    ocr: OCRResult,
    cleaned_display_path: str,
    original_copy_path: str,
    canonical_committed: bool,
) -> dict:
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "reviewStatus": "pending_human_review" if not canonical_committed else "approved_filename_pending_content_review",
        "generatedAt": now,
        "canonicalNamingCommitted": canonical_committed,
        "metadataConfidence": {
            "artist": round(proposal.artist_confidence, 3),
            "venue": round(proposal.venue_confidence, 3),
            "year": round(proposal.year_confidence, 3),
            "exactDate": round(proposal.date_confidence, 3),
            "coreMetadata": round(proposal.core_metadata_confidence, 3),
            "overall": round(proposal.overall_confidence, 3),
            "ocrAverage": round(ocr.average_confidence, 3),
            "lowConfidence": proposal.low_confidence,
            "canonicalEligible": proposal.canonical_eligible,
        },
        "ticket": draft_ticket,
        "proposedMetadata": {
            "artist": proposal.artist,
            "venue": proposal.venue,
            "year": proposal.year,
            "exactDate": proposal.exact_date,
            "city": proposal.city,
            "state": proposal.state,
            "country": proposal.country,
            "price": proposal.price,
        },
        "source": {
            "inputFile": source_path.name,
            "savedOriginal": original_copy_path,
            "savedCleaned": cleaned_display_path,
            "ocrConfig": ocr.best_config,
            "ocrPreview": ocr.lines[:30],
        },
        "warnings": proposal.warnings,
        "debug": {
            "artistCandidates": proposal.artist_candidates,
            "rejectedArtistCandidates": proposal.rejected_artist_candidates,
            "venueCandidates": proposal.venue_candidates,
            "dateCandidates": proposal.date_candidates,
            "filenamePriorsUsed": proposal.filename_priors_used,
            "filenamePriors": proposal.filename_priors,
            "approvalBlockers": proposal.approval_blockers,
        },
        "futureEnrichment": {
            "enabled": False,
            "notes": [
                "External venue/date verification can plug in after OCR review is approved.",
                "Ticketmaster/Setlist.fm/manual research can enrich city, lineup, and copy later without changing this draft contract.",
            ],
        },
    }


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def clean_lines(text: str) -> list[str]:
    return [normalize_space(line) for line in text.splitlines() if normalize_space(line)]


def extract_year(text: str) -> tuple[str, float]:
    matches = re.findall(r"\b(19[6-9]\d|20[0-3]\d)\b", text)
    if not matches:
        return "", 0.0
    counts: dict[str, int] = {}
    for match in matches:
        counts[match] = counts.get(match, 0) + 1
    year = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    confidence = 0.88 if counts[year] > 1 else 0.74
    return year, confidence


def extract_exact_date(text: str, candidates: list[dict] | None = None) -> tuple[str, float]:
    if candidates:
        top = candidates[0]
        return top["value"], top["score"]
    text_upper = text.upper()
    numeric = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](19[6-9]\d|20[0-3]\d)\b", text_upper)
    if numeric:
        month = numeric.group(1).zfill(2)
        day = numeric.group(2).zfill(2)
        year = numeric.group(3)
        return f"{year}-{month}-{day}", 0.82
    return "", 0.0


def extract_price(text: str) -> str:
    prices = re.findall(r"\$\s?\d+(?:\.\d{2})?", text)
    if prices:
        return prices[0].replace(" ", "")
    fallback = re.findall(r"\b\d{2,3}\.\d{2}\b", text)
    return f"${fallback[0]}" if fallback else ""


def extract_city_state(text: str, lines: list[str]) -> tuple[str, str]:
    for line in lines:
        match = CITY_STATE_PATTERN.search(line)
        if match:
            return match.group(1).strip(), match.group(2).strip()
    match = CITY_STATE_PATTERN.search(text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", ""


def extract_artist_candidates(lines: list[str], line_entries: list[dict]) -> tuple[list[dict], list[dict]]:
    candidates: list[dict] = []
    rejected: list[dict] = []

    for index, entry in enumerate(line_entries[:30] or fallback_line_entries(lines[:30])):
        line = entry["text"]
        evaluation = evaluate_artist_candidate(line, entry, index, lines)
        if evaluation["accepted"]:
            candidates.append(evaluation)
        else:
            rejected.append(evaluation)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    rejected.sort(key=lambda item: item["score"], reverse=True)
    return dedupe_candidates(candidates), rejected


def extract_venue_candidates(lines: list[str], line_entries: list[dict]) -> list[dict]:
    grouped: dict[str, dict[str, float | int]] = {}

    entries = line_entries or fallback_line_entries(lines)
    city_state_indices = [i for i, line in enumerate(lines) if CITY_STATE_PATTERN.search(line)]
    date_indices = [i for i, line in enumerate(lines) if extract_date_candidates(line)]

    for index, entry in enumerate(entries):
        line = entry["text"]
        lowered = line.lower()
        if not any(keyword in lowered for keyword in VENUE_KEYWORDS):
            continue

        normalized = normalize_venue(split_artist_prefix_from_venue(line))
        score = 0.62
        if normalized != normalize_venue(line):
            score += 0.10
        if len(normalized) <= 42:
            score += 0.10
        if line.isupper():
            score += 0.12
        if re.search(r"\b(at|@)\b", lowered):
            score += 0.08
        if re.search(r"\b(barclays|beacon|garden|center|centre|stadium|theatre|theater|ballroom|hall)\b", lowered):
            score += 0.08
        if near_indices(index, city_state_indices, 2) or near_indices(index, date_indices, 2):
            score += 0.10
        if has_address_neighbor(index, lines):
            score += 0.12

        if normalized not in grouped:
            grouped[normalized] = {"score": 0.0, "count": 0}
        grouped[normalized]["score"] = float(grouped[normalized]["score"]) + score
        grouped[normalized]["count"] = int(grouped[normalized]["count"]) + 1

    if not grouped:
        return []

    candidates: list[dict] = []
    for normalized, entry in grouped.items():
        repeated_bonus = min((int(entry["count"]) - 1) * 0.22, 0.44)
        score = float(entry["score"]) + repeated_bonus
        confidence = 0.72 + min(score / 3.6, 0.25)
        candidates.append({"value": normalized, "score": round(min(confidence, 0.97), 3)})

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return dedupe_candidates(candidates)


def normalize_artist(value: str) -> str:
    cleaned = normalize_space(value)
    cleaned = cleanup_ocr_wrapping_noise(cleaned)
    cleaned = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", cleaned)
    cleaned = cleaned.replace(" + ", " & ")
    if cleaned.isupper() or cleaned.islower():
        return cleanup_artist_noise_tokens(smart_title_case(cleaned))
    return cleanup_artist_noise_tokens(preserve_mixed_case(cleaned))


def normalize_venue(value: str) -> str:
    cleaned = normalize_space(value)
    cleaned = cleanup_ocr_wrapping_noise(cleaned)
    cleaned = re.sub(r"\s*,\s*[A-Z]{2}$", "", cleaned)
    cleaned = cleaned.strip(" '\"`“”‘’.,:;!-_")
    if cleaned.isupper() or cleaned.islower():
        return cleanup_venue_noise_tokens(smart_title_case(cleaned))
    return cleanup_venue_noise_tokens(preserve_mixed_case(cleaned))


def uppercase_ratio(value: str) -> float:
    letters = [char for char in value if char.isalpha()]
    if not letters:
        return 0.0
    uppercase = sum(1 for char in letters if char.isupper())
    return uppercase / len(letters)


def smart_title_case(value: str) -> str:
    tokens = re.split(r"(\s+|&|/|-)", value)
    output: list[str] = []

    for token in tokens:
        if not token:
            continue
        if token in {"&", "/", "-"} or token.isspace():
            output.append(token)
            continue
        lowered = token.lower()
        if lowered in SMART_TITLE_EXCEPTIONS:
            output.append(SMART_TITLE_EXCEPTIONS[lowered])
            continue
        output.append(titlecase_name_token(lowered))

    return preserve_mixed_case("".join(output).strip())


def titlecase_name_token(token: str) -> str:
    if token.startswith("mc") and len(token) > 2:
        return "Mc" + token[2:].capitalize()
    if token.startswith("mac") and len(token) > 3:
        return "Mac" + token[3:].capitalize()
    if "'" in token:
        return "'".join(part.capitalize() for part in token.split("'"))
    return token.capitalize()


def preserve_mixed_case(value: str) -> str:
    return re.sub(
        r"\bMc([a-z])",
        lambda match: "Mc" + match.group(1).upper(),
        value,
        flags=re.IGNORECASE,
    )


def build_line_entries(line_map: dict[tuple[int, int, int], dict]) -> list[dict]:
    raw_entries = []
    max_bottom = max((entry["bottom"] for entry in line_map.values()), default=1)
    for _, entry in sorted(line_map.items(), key=lambda item: (item[1]["top"], item[1]["left"])):
        text = normalize_space(" ".join(entry["words"]))
        if not text:
            continue
        avg_conf = sum(entry["confidences"]) / len(entry["confidences"]) if entry["confidences"] else 0.0
        raw_entries.append(
            {
                "text": text,
                "avg_confidence": round(max(avg_conf / 100.0, 0.0), 3),
                "top_ratio": round(entry["top"] / max_bottom, 3),
                "left": entry["left"],
                "top": entry["top"],
                "right": entry["right"],
                "bottom": entry["bottom"],
            }
        )
    return raw_entries


def fallback_line_entries(lines: list[str]) -> list[dict]:
    return [
        {"text": line, "avg_confidence": 0.0, "top_ratio": round(index / max(len(lines), 1), 3)}
        for index, line in enumerate(lines)
    ]


def top_candidate(candidates: list[dict]) -> tuple[str, float]:
    if not candidates:
        return "", 0.0
    return candidates[0]["value"], float(candidates[0]["score"])


def dedupe_candidates(candidates: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []
    for candidate in candidates:
        key = slugify(candidate["value"])
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def extract_date_candidates(text: str) -> list[dict]:
    text_upper = text.upper()
    month_pattern = "|".join(MONTHS.keys())
    candidates: list[dict] = []

    for match in re.finditer(
        rf"\b(?:MON|TUE|WED|THU|THUR|FRI|SAT|SUN|MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)?\s*({month_pattern})\s+(\d{{1,2}}),?\s+(19[6-9]\d|20[0-3]\d)\b",
        text_upper,
    ):
        month = MONTHS[match.group(1)]
        day = match.group(2).zfill(2)
        year = match.group(3)
        candidates.append(
            {
                "value": f"{year}-{month}-{day}",
                "score": 0.92,
                "source": match.group(0).strip(),
            }
        )

    for match in re.finditer(r"\b(\d{1,2})[/-](\d{1,2})[/-](19[6-9]\d|20[0-3]\d)\b", text_upper):
        month = match.group(1).zfill(2)
        day = match.group(2).zfill(2)
        year = match.group(3)
        candidates.append(
            {
                "value": f"{year}-{month}-{day}",
                "score": 0.82,
                "source": match.group(0).strip(),
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return dedupe_candidates(candidates)


def build_filename_priors(source_path: Path | None) -> dict:
    if source_path is None:
        return empty_filename_priors()

    stem = slugify(source_path.stem)
    parts = [part for part in stem.split("-") if part]
    if len(parts) < 3 or not re.fullmatch(r"(?:19|20)\d{2}", parts[-1]):
        return empty_filename_priors()

    year = parts[-1]
    body = parts[:-1]
    best_split = 1
    best_score = -1.0

    for split in range(1, len(body)):
        artist_tokens = body[:split]
        venue_tokens = body[split:]
        venue_text = " ".join(venue_tokens)
        score = 0.0
        if any(keyword in venue_text for keyword in VENUE_KEYWORDS):
            score += 1.0
        if 1 <= len(venue_tokens) <= 4:
            score += 0.35
        if 1 <= len(artist_tokens) <= 6:
            score += 0.2
        if venue_tokens[-1] in {"lunch", "center", "hall", "theater", "theatre", "arena", "ballroom", "stadium", "club"}:
            score += 0.4
        if (
            len(artist_tokens) == 4
            and len(venue_tokens) == 2
            and venue_tokens[-1] in {"center", "hall", "theater", "theatre", "arena", "lunch", "club"}
        ):
            score += 0.85
        if (
            len(venue_tokens) >= 4
            and looks_like_slug_name_pair(venue_tokens[:2])
            and any(token in VENUE_KEYWORDS for token in venue_tokens[2:])
        ):
            score -= 0.85
        if score > best_score:
            best_score = score
            best_split = split

    artist_tokens = body[:best_split]
    venue_tokens = body[best_split:]
    artist_parts: list[str] = []
    artist_value = format_slug_tokens_as_name(artist_tokens)
    if len(artist_tokens) == 4 and all(len(token) <= 10 for token in artist_tokens):
        first_artist = format_slug_tokens_as_name(artist_tokens[:2])
        second_artist = format_slug_tokens_as_name(artist_tokens[2:])
        artist_parts = [first_artist, second_artist]
        artist_value = f"{first_artist} & {second_artist}"
    elif artist_value:
        artist_parts = [artist_value]

    venue_value = format_slug_tokens_as_name(venue_tokens)
    structured_plausible = bool(
        artist_value
        and venue_value
        and year
        and (len(artist_parts) == 2 or len(artist_tokens) in {1, 2, 3, 4})
        and 1 <= len(venue_tokens) <= 3
    )

    return {
        "used": False,
        "artist": artist_value,
        "artistParts": artist_parts,
        "venue": venue_value,
        "venueParts": [venue_value] if venue_value else [],
        "year": year,
        "structuredPlausible": structured_plausible,
        "sourceStem": stem,
        "bodyParts": body,
        "bestSplit": best_split,
    }


def empty_filename_priors() -> dict:
    return {
        "used": False,
        "artist": "",
        "artistParts": [],
        "venue": "",
        "venueParts": [],
        "year": "",
        "structuredPlausible": False,
        "sourceStem": "",
        "bodyParts": [],
        "bestSplit": 0,
    }


def format_slug_tokens_as_name(tokens: list[str]) -> str:
    formatted: list[str] = []
    for token in tokens:
        if not token:
            continue
        if len(token) <= 2 and token.isalpha():
            formatted.append(token.upper())
        else:
            formatted.append(titlecase_name_token(token.lower()))
    return " ".join(formatted).strip()


def looks_like_slug_name_pair(tokens: list[str]) -> bool:
    if len(tokens) != 2:
        return False
    return all(re.fullmatch(r"[a-z]{2,12}", token) for token in tokens)


def has_embedded_artist_tokens_in_venue(value: str, artist_parts: list[str]) -> bool:
    if not value or not artist_parts:
        return False
    lowered_value = value.lower()
    for part in artist_parts:
        tokens = [token.lower() for token in part.split() if token]
        if not tokens:
            continue
        if all(token in lowered_value for token in tokens):
            return True
    return False


def near_indices(index: int, indices: list[int], distance: int) -> bool:
    return any(abs(index - candidate) <= distance for candidate in indices)


def has_address_neighbor(index: int, lines: list[str]) -> bool:
    window = lines[max(0, index - 2): min(len(lines), index + 3)]
    return any(
        re.search(r"\b\d{2,5}\b", line) or CITY_STATE_PATTERN.search(line)
        for line in window
    )


def evaluate_artist_candidate(line: str, entry: dict, index: int, all_lines: list[str]) -> dict:
    lowered = line.lower()
    reasons: list[str] = []
    rejection_reasons: list[str] = []
    normalized_line = normalize_space(line)
    top_ratio = float(entry.get("top_ratio", index / max(len(all_lines), 1)))
    alpha_chars = sum(char.isalpha() for char in normalized_line)
    digit_chars = sum(char.isdigit() for char in normalized_line)
    non_letter_chars = sum(not char.isalpha() and not char.isspace() and char not in {"&", "'"} for char in normalized_line)
    words = [word for word in re.split(r"\s+", normalized_line) if word]
    alpha_ratio = alpha_chars / max(len(normalized_line.replace(" ", "")), 1)
    has_venue_keyword = any(keyword in lowered for keyword in VENUE_KEYWORDS)

    if len(normalized_line) < 3 or len(normalized_line) > 70:
        rejection_reasons.append("length out of artist range")
    if any(term in lowered for term in NEGATIVE_ARTIST_TERMS):
        rejection_reasons.append("contains common ticket-fee or seating term")
    if any(stop in lowered for stop in ARTIST_STOP_WORDS):
        rejection_reasons.append("contains ticket metadata label")
    if "with " in lowered:
        rejection_reasons.append("looks like opener/support billing")
    if re.search(r"\b(event code|section|row|seat|admission)\b", lowered):
        rejection_reasons.append("near seating or event label")
    if re.search(r"\$\s?\d", normalized_line) or re.search(r"\b\d{1,2}\.\d{2}\b", normalized_line):
        rejection_reasons.append("contains price-like text")
    if digit_chars > 0:
        rejection_reasons.append("contains digits")
    if re.fullmatch(r"[A-Za-z]*\d+[A-Za-z0-9]*", normalized_line.replace(" ", "")):
        rejection_reasons.append("looks like alphanumeric event code")
    if re.fullmatch(r"[A-Z0-9]{5,}", normalized_line.replace(" ", "")):
        rejection_reasons.append("looks like uppercase code token")
    if non_letter_chars > 3:
        rejection_reasons.append("contains too many non-letter characters")

    split_artist = split_mixed_artist_venue(normalized_line)
    if has_venue_keyword:
        if split_artist and split_artist != normalized_line:
            reasons.append("split mixed artist/venue line")
            normalized_line = split_artist
            lowered = normalized_line.lower()
            words = [word for word in re.split(r"\s+", normalized_line) if word]
            alpha_chars = sum(char.isalpha() for char in normalized_line)
            digit_chars = sum(char.isdigit() for char in normalized_line)
            alpha_ratio = alpha_chars / max(len(normalized_line.replace(" ", "")), 1)
            has_venue_keyword = any(keyword in lowered for keyword in VENUE_KEYWORDS)
        else:
            rejection_reasons.append("contains venue keyword")

    if not words or len(words) > 6:
        rejection_reasons.append("word count is not plausible for an artist line")
    if alpha_ratio < 0.78:
        rejection_reasons.append("not dominated by alphabetic characters")
    if len(words) == 1 and len(words[0]) < 4 and not looks_like_short_band_name(words[0]):
        rejection_reasons.append("too short to be a likely artist")
    if not looks_like_name_shape(normalized_line):
        rejection_reasons.append("does not resemble a recognizable artist name")

    plausibility = 0.0
    if not rejection_reasons:
        center_bias = 1.0 - abs(top_ratio - 0.32)
        plausibility = 0.30 + max(center_bias, 0.0) * 0.24
        plausibility += 0.14 if len(words) <= 4 else 0.0
        plausibility += 0.12 if uppercase_ratio(normalized_line) > 0.72 else 0.0
        plausibility += 0.14 if ("&" in normalized_line or " and " in lowered) else 0.0
        plausibility += 0.10 if alpha_ratio > 0.92 else 0.0
        plausibility += 0.08 if looks_like_name_or_band(normalized_line) else 0.0
        plausibility += 0.10 if len(words) == 1 and looks_like_short_band_name(words[0]) else 0.0
        plausibility += 0.08 if looks_like_initial_name_pair(normalized_line) else 0.0
        plausibility -= min(non_letter_chars * 0.06, 0.18)
        if lowered.startswith("the "):
            plausibility += 0.04
        reasons.append("looks like a human-readable artist/band line")

    accepted = not rejection_reasons and plausibility >= 0.58
    if not accepted and plausibility < 0.58 and not rejection_reasons:
        rejection_reasons.append("artist plausibility below acceptance threshold")

    normalized_value = normalize_artist(normalized_line)
    return {
        "accepted": accepted,
        "value": normalized_value,
        "score": round(min(plausibility, 0.97), 3),
        "plausibility": round(min(plausibility, 0.97), 3),
        "ocrConfidence": round(float(entry.get("avg_confidence", 0.0)), 3),
        "topRatio": round(top_ratio, 3),
        "sourceLine": line,
        "reasons": reasons if accepted else rejection_reasons,
    }


def split_mixed_artist_venue(line: str) -> str:
    tokens = line.split()
    lowered_tokens = [token.lower() for token in tokens]
    for index, token in enumerate(lowered_tokens):
        if token in VENUE_KEYWORDS:
            prefix = tokens[:max(index - 2, 1)]
            if prefix:
                return " ".join(prefix)
    return line


def split_artist_prefix_from_venue(line: str) -> str:
    tokens = line.split()
    lowered_tokens = [token.lower() for token in tokens]
    venue_indexes = [index for index, token in enumerate(lowered_tokens) if token in VENUE_KEYWORDS]
    if not venue_indexes:
        return line
    venue_index = venue_indexes[0]
    for offset in (2, 1, 0):
        start = max(venue_index - offset, 0)
        candidate = " ".join(tokens[start:])
        prefix = " ".join(tokens[:start])
        if not candidate:
            continue
        if prefix and looks_like_name_shape(prefix):
            return candidate
        if candidate.lower() in {
            "beacon theater",
            "beacon theatre",
            "frank erwin center",
            "music hall",
            "liberty lunch",
        }:
            return candidate
    return line


def looks_like_name_or_band(value: str) -> bool:
    if re.fullmatch(r"[A-Za-z]+(?:\s+[A-Za-z&']+){0,4}", value):
        return True
    if re.fullmatch(r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4}", value):
        return True
    return False


def looks_like_name_shape(value: str) -> bool:
    tokens = [token.strip(".,'\"") for token in value.split() if token]
    if not 1 <= len(tokens) <= 6:
        return False
    valid_tokens = 0
    for token in tokens:
        if re.fullmatch(r"[A-Za-z]{2,12}", token):
            valid_tokens += 1
        elif re.fullmatch(r"[A-Za-z]{1,3}", token) and token.isupper():
            valid_tokens += 1
    return valid_tokens >= max(1, len(tokens) - 1)


def looks_like_initial_name_pair(value: str) -> bool:
    tokens = [token.strip(".,'\"") for token in value.split() if token]
    if len(tokens) < 2:
        return False
    return any(re.fullmatch(r"[A-Z]{2,3}", token) for token in tokens[:2])


def detect_multi_artist_billing(lines: list[str]) -> tuple[str, float]:
    strong_lines = []
    for line in lines[:12]:
        cleaned = normalize_artist(line)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if any(keyword in lowered for keyword in VENUE_KEYWORDS):
            continue
        if any(term in lowered for term in NEGATIVE_ARTIST_TERMS):
            continue
        if re.search(r"\d", cleaned):
            continue
        if looks_like_name_shape(cleaned):
            strong_lines.append(cleaned)

    for index in range(len(strong_lines) - 1):
        first = strong_lines[index]
        second = strong_lines[index + 1]
        if first == second:
            continue
        if looks_like_partial_artist_name(first) and looks_like_partial_artist_name(second):
            return f"{first} & {second}", 0.78

    for line in strong_lines:
        upper = line.upper()
        if " / " in line:
            parts = [normalize_artist(part) for part in line.split(" / ") if normalize_artist(part)]
            if len(parts) == 2:
                return f"{parts[0]} & {parts[1]}", 0.82
        if " + " in line:
            parts = [normalize_artist(part) for part in line.split(" + ") if normalize_artist(part)]
            if len(parts) == 2:
                return f"{parts[0]} & {parts[1]}", 0.82
        if " AND " in upper:
            parts = [normalize_artist(part) for part in re.split(r"\bAND\b", line, flags=re.IGNORECASE) if normalize_artist(part)]
            if len(parts) == 2:
                return f"{parts[0]} & {parts[1]}", 0.8

    return "", 0.0


def looks_like_partial_artist_name(value: str) -> bool:
    tokens = [token for token in value.split() if token]
    if not 1 <= len(tokens) <= 2:
        return False
    return all(
        re.fullmatch(r"[A-Za-z]{2,12}", token) or re.fullmatch(r"[A-Z]{2,3}", token)
        for token in tokens
    )


def looks_like_short_band_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]{4,6}", value))


def compute_core_metadata_confidence(
    artist_confidence: float,
    venue_confidence: float,
    year_confidence: float,
) -> float:
    weighted = (
        (artist_confidence * 0.40)
        + (venue_confidence * 0.34)
        + (year_confidence * 0.26)
    )
    if artist_confidence >= 0.62 and venue_confidence >= 0.60 and year_confidence >= 0.70:
        weighted = max(weighted, 0.74)
    if min(artist_confidence, venue_confidence, year_confidence) < 0.5:
        weighted = min(weighted, 0.58)
    return min(weighted, 0.99)


def build_approval_blockers(
    *,
    artist: str,
    venue: str,
    year: str,
    artist_confidence: float,
    venue_confidence: float,
    year_confidence: float,
    core_metadata_confidence: float,
) -> list[str]:
    blockers: list[str] = []
    if not artist:
        blockers.append("artist missing")
    if not venue:
        blockers.append("venue missing")
    if not year:
        blockers.append("year missing")
    if artist_confidence < 0.54:
        blockers.append("artist confidence too weak")
    if venue_confidence < 0.54:
        blockers.append("venue confidence too weak")
    if year_confidence < 0.54:
        blockers.append("year confidence too weak")
    if core_metadata_confidence < 0.64:
        blockers.append("core metadata confidence below approval threshold")
    return blockers


def looks_like_short_clean_venue(value: str) -> bool:
    words = [word for word in value.split() if word]
    if not 1 <= len(words) <= 3:
        return False
    return bool(re.fullmatch(r"[A-Za-z]+(?:\s+[A-Za-z]+){0,2}", value))


def cleanup_ocr_wrapping_noise(value: str) -> str:
    cleaned = value.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.strip(string.punctuation + " ")


def cleanup_artist_noise_tokens(value: str) -> str:
    tokens = [token for token in value.split() if token]
    if len(tokens) < 2:
        return value

    strong_indices = [
        index for index, token in enumerate(tokens)
        if looks_like_short_band_name(token) or re.fullmatch(r"[A-Z][a-z]{3,12}", token) or token.lower() in {"blur", "beck", "queen"}
    ]
    if len(strong_indices) == 1:
        index = strong_indices[0]
        token = tokens[index]
        left_noise = all(len(part) <= 3 for part in tokens[:index])
        right_noise = all(len(part) <= 3 for part in tokens[index + 1:])
        if left_noise and right_noise:
            return smart_title_case(token)

    filtered = [token for token in tokens if len(token) > 2 or token in {"&", "and", "The"}]
    if filtered:
        return " ".join(filtered)
    return value


def cleanup_venue_noise_tokens(value: str) -> str:
    cleaned = value.strip(" '\"`“”‘’.,:;!-_")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned
