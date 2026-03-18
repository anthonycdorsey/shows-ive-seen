from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

from approved_ticket_to_site import prepare_site_write_plan, write_site_write_plan
from process_ticket import ensure_dirs
from site_writer_support import build_share_description, build_share_title, slugify
from ticket_enrichment_support import (
    load_external_facts,
    load_writer_context,
    prepare_enrichment_plan,
)
from ticket_metadata import (
    build_draft_ticket,
    build_review_payload,
    build_review_slug,
    configure_tesseract,
    propose_metadata,
    run_ocr_variants,
)
from ticket_pipeline import (
    build_contact_sheet,
    clean_ticket_image,
    list_supported_images,
    load_image,
    save_jpeg,
)


REVIEW_FIELDS = [
    ("artist", "artist"),
    ("venue", "venue"),
    ("city", "city"),
    ("state", "state"),
    ("country", "country"),
    ("year", "year"),
    ("exactDate", "exactDate"),
    ("price", "price"),
    ("copy", "copy"),
    ("extendedNotes", "extendedNotes"),
    ("tags", "tags"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified Shows I Saw pipeline: OCR, enrichment, review, edit, and publish."
    )
    parser.add_argument(
        "--archive-dir",
        default=str(Path(__file__).resolve().parent),
        help="Archive ingest root. Defaults to the folder containing this script.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        default=None,
        help="Optional full path to tesseract.exe.",
    )
    parser.add_argument(
        "--external-facts",
        default=None,
        help="Optional local JSON file containing factual enrichment records keyed by slug or filename stem.",
    )
    return parser.parse_args()


def prompt_with_default(label: str, default: str = "") -> str:
    if default:
        value = input(f"{label} [{default}]: ").strip()
        return value if value else default
    return input(f"{label}: ").strip()


def prompt_yes_edit_skip() -> str:
    response = input("\nApprove? [y/edit/skip]: ").strip().lower()
    if response in {"y", "yes"}:
        return "y"
    if response == "edit":
        return "edit"
    return "skip"


def move_source(source_path: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    candidate = destination_dir / source_path.name
    if candidate.exists():
        suffix = perf_counter_ns_suffix()
        candidate = destination_dir / f"{source_path.stem}-{suffix}{source_path.suffix}"
    shutil.move(str(source_path), str(candidate))
    return candidate


def perf_counter_ns_suffix() -> str:
    return str(int(perf_counter() * 1000000))


def persist_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_ticket_fields(ticket: dict) -> dict:
    normalized = dict(ticket)
    artist = normalized.get("artist", "").strip()
    venue = normalized.get("venue", "").strip()
    year = normalized.get("year", "").strip()

    normalized["artist"] = artist
    normalized["venue"] = venue
    normalized["city"] = normalized.get("city", "").strip()
    normalized["state"] = normalized.get("state", "").strip()
    normalized["country"] = (normalized.get("country", "") or "USA").strip()
    normalized["year"] = year
    normalized["exactDate"] = normalized.get("exactDate", "").strip()
    normalized["price"] = normalized.get("price", "").strip()
    normalized["copy"] = normalized.get("copy", "").strip()
    normalized["extendedNotes"] = normalized.get("extendedNotes", "").strip()
    normalized["youtubeUrl"] = normalized.get("youtubeUrl", "").strip()
    normalized["rotation"] = normalized.get("rotation", "") or "0deg"
    normalized["companions"] = ensure_list(normalized.get("companions", []))
    normalized["photos"] = ensure_list(normalized.get("photos", []))
    normalized["tags"] = ensure_list(normalized.get("tags", []))

    artist_slug = slugify(artist)
    venue_slug = slugify(venue)
    normalized["artistSlug"] = artist_slug
    if artist_slug and venue_slug and year:
        slug = f"{artist_slug}-{venue_slug}-{year}"
        normalized["slug"] = slug
        normalized["img"] = f"{slug}.jpg"
        normalized["shareImage"] = f"{slug}-share.jpg"
    else:
        normalized["slug"] = normalized.get("slug", "").strip()
        normalized["img"] = normalized.get("img", "").strip()
        normalized["shareImage"] = normalized.get("shareImage", "").strip()

    if normalized.get("artist") and normalized.get("venue") and normalized.get("year"):
        normalized["shareTitle"] = build_share_title(normalized)
    if not normalized.get("shareDescription") and normalized.get("copy"):
        normalized["shareDescription"] = build_share_description(normalized, normalized["copy"])

    return normalized


def ensure_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def print_review(ticket: dict) -> None:
    print("\nProposed record\n")
    for label, field in REVIEW_FIELDS:
        value = ticket.get(field, "")
        if isinstance(value, list):
            value = ", ".join(value)
        print(f"{label}: {value or '[blank]'}")


def edit_ticket(ticket: dict) -> dict:
    edited = dict(ticket)
    edited["artist"] = prompt_with_default("artist", edited.get("artist", ""))
    edited["venue"] = prompt_with_default("venue", edited.get("venue", ""))
    edited["city"] = prompt_with_default("city", edited.get("city", ""))
    edited["state"] = prompt_with_default("state", edited.get("state", ""))
    edited["country"] = prompt_with_default("country", edited.get("country", "USA"))
    edited["year"] = prompt_with_default("year", edited.get("year", ""))
    edited["exactDate"] = prompt_with_default("exactDate", edited.get("exactDate", ""))
    edited["price"] = prompt_with_default("price", edited.get("price", ""))
    edited["copy"] = prompt_with_default("copy", edited.get("copy", ""))
    edited["extendedNotes"] = prompt_with_default("extendedNotes", edited.get("extendedNotes", ""))
    edited["tags"] = ensure_list(prompt_with_default("tags (comma-separated)", ", ".join(edited.get("tags", []))))
    edited["companions"] = ensure_list(prompt_with_default("companions (comma-separated)", ", ".join(edited.get("companions", []))))
    edited["photos"] = ensure_list(prompt_with_default("photos (comma-separated)", ", ".join(edited.get("photos", []))))
    edited["youtubeUrl"] = prompt_with_default("youtubeUrl", edited.get("youtubeUrl", ""))
    return normalize_ticket_fields(edited)


def artifact_base_name(ticket: dict, source_path: Path, review_slug: str) -> str:
    return ticket.get("slug") or review_slug or source_path.stem


def save_review_artifacts(
    *,
    source_path: Path,
    archive_dir: Path,
    paths: dict[str, Path],
    ticket: dict,
    payload: dict,
    ocr,
    cleaned_result,
    original_image,
    review_slug: str,
) -> dict[str, Path]:
    base_name = artifact_base_name(ticket, source_path, review_slug)
    working_original = paths["review"] / f"{base_name}-original.jpg"
    ocr_working = paths["ocr_working"] / f"{base_name}-ocr-working.jpg"
    contact_sheet_output = paths["contact_sheets"] / f"{base_name}-contact-sheet.jpg"
    ocr_text_output = paths["ocr"] / f"{base_name}-ocr.txt"
    ocr_debug_output = paths["ocr"] / f"{base_name}-ocr-debug.json"
    notes_output = paths["notes"] / f"{base_name}.txt"
    draft_path = paths["draft_json"] / f"{base_name}.json"

    shutil.copy2(source_path, working_original)
    save_jpeg(cleaned_result.display_image, ocr_working)
    contact_sheet = build_contact_sheet(
        original=original_image,
        cleaned=cleaned_result.display_image,
        title=f"Shows I Saw Review: {base_name}",
    )
    save_jpeg(contact_sheet, contact_sheet_output)

    ocr_text = "\n".join(
        [
            "SHOWS I SAW - OCR DIAGNOSTICS",
            "",
            f"Selected best variant: {ocr.best_variant_label}",
            f"Selected config: {ocr.best_config}",
            "",
            "RAW OCR TEXT",
            "",
            ocr.text,
            "",
            "CLEANED OCR TEXT",
            "",
            ocr.cleaned_text,
        ]
    )
    ocr_text_output.write_text(ocr_text, encoding="utf-8")
    ocr_debug_output.write_text(
        json.dumps(
            {
                "selectedBestVariant": ocr.best_variant_label,
                "selectedConfig": ocr.best_config,
                "averageConfidence": round(ocr.average_confidence, 3),
                "topArtistCandidates": payload.get("debug", {}).get("artistCandidates", []),
                "rejectedArtistCandidates": payload.get("debug", {}).get("rejectedArtistCandidates", []),
                "topVenueCandidates": payload.get("debug", {}).get("venueCandidates", []),
                "dateCandidates": payload.get("debug", {}).get("dateCandidates", []),
                "runs": ocr.debug_runs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    notes_output.write_text(
        "\n".join(
            [
                "SHOWS I SAW - UNIFIED PIPELINE REVIEW",
                "",
                f"source: {source_path}",
                f"draft: {draft_path}",
                f"ocr working: {ocr_working}",
                f"contact sheet: {contact_sheet_output}",
                f"ocr debug: {ocr_debug_output}",
                f"slug: {ticket.get('slug', '')}",
            ]
        ),
        encoding="utf-8",
    )
    persist_payload(draft_path, payload)
    return {
        "draft": draft_path,
        "ocrWorking": ocr_working,
        "contactSheet": contact_sheet_output,
        "ocrText": ocr_text_output,
        "ocrDebug": ocr_debug_output,
        "notes": notes_output,
        "originalReviewCopy": working_original,
    }


def build_unified_payload(
    *,
    source_path: Path,
    source_saved_original: str,
    proposal,
    ticket: dict,
    ocr,
    cleaned_display_path: str,
    canonical_committed: bool,
) -> dict:
    payload = build_review_payload(
        source_path=source_path,
        draft_ticket=ticket,
        proposal=proposal,
        ocr=ocr,
        cleaned_display_path=cleaned_display_path,
        original_copy_path=source_saved_original,
        canonical_committed=canonical_committed,
    )
    payload["ticket"] = ticket
    payload["reviewStatus"] = "approved_filename_pending_content_review" if canonical_committed else "pending_human_review"
    payload["source"]["savedOriginal"] = source_saved_original
    payload["source"]["sourceOriginal"] = source_saved_original
    payload["source"]["selectedIncomingFile"] = source_path.name
    payload["reviewArtifacts"] = {
        "finalImageFilename": ticket.get("img", ""),
        "finalShareImageFilename": ticket.get("shareImage", ""),
        "suggestedShareFolder": f"share/{ticket.get('slug', '')}/" if ticket.get("slug") else "",
        "suggestedSharePage": f"share/{ticket.get('slug', '')}/index.html" if ticket.get("slug") else "",
    }
    return payload


def refresh_payload_from_ticket(payload: dict, ticket: dict, source_path: Path, canonical_committed: bool) -> dict:
    payload["ticket"] = ticket
    payload["canonicalNamingCommitted"] = canonical_committed
    payload["reviewStatus"] = "approved_filename_pending_content_review" if canonical_committed else "pending_human_review"
    payload.setdefault("source", {})
    payload["source"]["savedOriginal"] = str(source_path)
    payload["source"]["sourceOriginal"] = str(source_path)
    payload["source"]["selectedIncomingFile"] = source_path.name
    payload["reviewArtifacts"] = {
        "finalImageFilename": ticket.get("img", ""),
        "finalShareImageFilename": ticket.get("shareImage", ""),
        "suggestedShareFolder": f"share/{ticket.get('slug', '')}/" if ticket.get("slug") else "",
        "suggestedSharePage": f"share/{ticket.get('slug', '')}/index.html" if ticket.get("slug") else "",
    }
    return payload


def process_image(
    source_path: Path,
    *,
    archive_dir: Path,
    paths: dict[str, Path],
    writer_context: dict,
    external_facts: dict,
) -> tuple[bool, dict]:
    print(f"\n=== {source_path.name} ===")
    original_image = load_image(source_path)
    cleaned_result = clean_ticket_image(original_image)
    ocr = run_ocr_variants(
        [
            ("display", cleaned_result.display_image),
            ("ocr-gray", cleaned_result.ocr_gray_image),
            ("ocr-binary", cleaned_result.ocr_binary_image),
        ],
        source_path=source_path,
    )
    proposal = propose_metadata(ocr, source_path=source_path)
    review_slug = build_review_slug(source_path)
    ticket = normalize_ticket_fields(build_draft_ticket(proposal))
    payload = build_unified_payload(
        source_path=source_path,
        source_saved_original=str(source_path),
        proposal=proposal,
        ticket=ticket,
        ocr=ocr,
        cleaned_display_path="",
        canonical_committed=False,
    )
    artifact_paths = save_review_artifacts(
        source_path=source_path,
        archive_dir=archive_dir,
        paths=paths,
        ticket=ticket,
        payload=payload,
        ocr=ocr,
        cleaned_result=cleaned_result,
        original_image=original_image,
        review_slug=review_slug,
    )
    payload["source"]["savedCleaned"] = str(artifact_paths["ocrWorking"])
    persist_payload(artifact_paths["draft"], payload)

    plan = prepare_enrichment_plan(
        artifact_paths["draft"],
        archive_dir=archive_dir,
        writer_context=writer_context,
        external_facts=external_facts,
    )
    payload = plan.enriched_payload
    persist_payload(artifact_paths["draft"], payload)

    while True:
        ticket = normalize_ticket_fields(payload.get("ticket", {}))
        payload = refresh_payload_from_ticket(payload, ticket, source_path, canonical_committed=False)
        print_review(ticket)
        action = prompt_yes_edit_skip()

        if action == "edit":
            ticket = edit_ticket(ticket)
            payload = refresh_payload_from_ticket(payload, ticket, source_path, canonical_committed=False)
            persist_payload(artifact_paths["draft"], payload)
            plan = prepare_enrichment_plan(
                artifact_paths["draft"],
                archive_dir=archive_dir,
                writer_context=writer_context,
                external_facts=external_facts,
            )
            payload = plan.enriched_payload
            persist_payload(artifact_paths["draft"], payload)
            continue

        if action == "skip":
            moved = move_source(source_path, paths["rejected"])
            print(f"\nsource file moved to: rejected/ ({moved.name})")
            print("canonical original stored at: [not written]")
            print("site image stored at: [not written]")
            return False, {"sourceMoved": str(moved), "status": "rejected"}

        ticket = normalize_ticket_fields(payload.get("ticket", {}))
        payload = refresh_payload_from_ticket(payload, ticket, source_path, canonical_committed=True)
        persist_payload(artifact_paths["draft"], payload)

        publish_plan, publish_error = prepare_site_write_plan(artifact_paths["draft"], archive_dir=archive_dir)
        if publish_plan is None:
            moved = move_source(source_path, paths["rejected"])
            print("\nPublish blocked\n")
            print(publish_error or "Unknown publish error")
            print(f"\nsource file moved to: rejected/ ({moved.name})")
            print("canonical original stored at: [not written]")
            print("site image stored at: [not written]")
            return False, {"sourceMoved": str(moved), "status": "blocked", "error": publish_error or ""}

        write_site_write_plan(publish_plan)
        moved = move_source(source_path, paths["processed"])
        print("\nTicket published successfully\n")
        print(f"Slug: {publish_plan.site_ticket.get('slug', '')}")
        print(f"source file moved to: processed/ ({moved.name})")
        print(f"canonical original stored at: {publish_plan.final_image_source}")
        print(f"site image stored at: {publish_plan.final_site_image}")
        return True, {
            "sourceMoved": str(moved),
            "status": "published",
            "slug": publish_plan.site_ticket.get("slug", ""),
        }


def main() -> None:
    args = parse_args()
    archive_dir = Path(args.archive_dir).resolve()
    paths = ensure_dirs(archive_dir)
    paths["review"] = paths["needs_review"]
    configure_tesseract(args.tesseract_cmd)

    incoming_images = list_supported_images(paths["incoming"])
    if not incoming_images:
        print("\nNo images found in incoming/.\n")
        return

    external_facts = load_external_facts(Path(args.external_facts).resolve()) if args.external_facts else {}
    writer_context = load_writer_context(archive_dir)

    print("\nShows I Saw - Unified Pipeline\n")
    print(f"Found {len(incoming_images)} image(s) in incoming/.\n")

    published = 0
    rejected = 0
    for source_path in incoming_images:
        success, result = process_image(
            source_path,
            archive_dir=archive_dir,
            paths=paths,
            writer_context=writer_context,
            external_facts=external_facts,
        )
        if success:
            published += 1
            writer_context = load_writer_context(archive_dir)
        else:
            rejected += 1

    print("\nSummary\n")
    print(f"published: {published}")
    print(f"rejected:  {rejected}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        sys.exit(1)
