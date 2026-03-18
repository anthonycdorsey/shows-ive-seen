from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime
from time import perf_counter

from ticket_metadata import (
    build_review_slug,
    build_draft_ticket,
    build_review_payload,
    configure_tesseract,
    propose_metadata,
    run_ocr_variants,
)
from approved_ticket_to_site import prepare_site_write_plan, print_site_write_preview, write_site_write_plan
from ticket_pipeline import (
    build_contact_sheet,
    clean_ticket_image,
    list_supported_images,
    load_image,
    save_jpeg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process one already-cropped single-ticket image into a reviewable draft package."
    )
    parser.add_argument("image", nargs="?", help="Path to a cropped JPG ticket image.")
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
        "--yes",
        action="store_true",
        help="Approve canonical filenames automatically when confidence is high.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing approved canonical files without prompting.",
    )
    return parser.parse_args()


def choose_input_file(incoming_dir: Path) -> Path:
    candidates = list_supported_images(incoming_dir)
    if not candidates:
        raise FileNotFoundError(f"No supported images found in {incoming_dir}")

    print("\nAvailable files in incoming/\n")
    for index, path in enumerate(candidates, start=1):
        print(f"{index}. {path.name}")

    while True:
        choice = input("\nChoose a file number: ").strip()
        if choice.isdigit():
            selected = int(choice)
            if 1 <= selected <= len(candidates):
                return candidates[selected - 1]
        print("Please enter a valid number.")


def confirm(prompt: str) -> bool:
    response = input(f"{prompt} [y/N]: ").strip().lower()
    return response in {"y", "yes"}


def canonical_paths_available(*candidate_paths: Path) -> bool:
    return all(not path.exists() for path in candidate_paths)


def ensure_dirs(archive_dir: Path) -> dict[str, Path]:
    paths = {
        "incoming": archive_dir / "incoming",
        "processed": archive_dir / "processed",
        "rejected": archive_dir / "rejected",
        "originals": archive_dir / "originals",
        "original_copy": archive_dir / "originals" / "original-copy",
        "ocr_working": archive_dir / "working" / "ocr-support",
        "published": archive_dir / "published",
        "draft_json": archive_dir / "review" / "draft-json",
        "notes": archive_dir / "review" / "notes",
        "contact_sheets": archive_dir / "review" / "contact-sheets",
        "needs_review": archive_dir / "review" / "needs-review",
        "ocr": archive_dir / "review" / "ocr",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def move_source_out_of_incoming(source_path: Path, incoming_dir: Path, destination_dir: Path) -> Path:
    if source_path.parent.resolve() != incoming_dir.resolve():
        return source_path

    candidate = destination_dir / source_path.name
    if candidate.exists():
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        candidate = destination_dir / f"{source_path.stem}-{timestamp}{source_path.suffix}"
    shutil.move(str(source_path), str(candidate))
    return candidate


def build_notes_text(
    source_path: Path,
    original_output: Path,
    final_image_output: Path,
    ocr_working_output: Path,
    json_output: Path,
    contact_sheet_output: Path,
    proposal,
    ocr,
    image_notes: list[str],
    canonical_committed: bool,
    review_slug: str,
) -> str:
    next_step = "Review metadata/content and manually publish later." if canonical_committed else "Resolve metadata uncertainty and rename only after approval."
    return f"""SHOWS I SAW - SINGLE TICKET INGEST REVIEW

Selected source:
{source_path}

Saved source archive copy:
{original_output}

Final canonical image uses original (no enhancement applied):
{final_image_output}

Temporary OCR working image:
{ocr_working_output}

Draft JSON:
{json_output}

Contact sheet:
{contact_sheet_output}

Proposed metadata:
- artist: {proposal.artist}
- venue: {proposal.venue}
- year: {proposal.year}
- exactDate: {proposal.exact_date}
- city: {proposal.city}
- state: {proposal.state}
- price: {proposal.price}
- slug: {proposal.slug}
- reviewSlug: {review_slug}

Confidence:
- OCR average: {ocr.average_confidence:.2f}
- artist: {proposal.artist_confidence:.2f}
- venue: {proposal.venue_confidence:.2f}
- year: {proposal.year_confidence:.2f}
- exact date: {proposal.date_confidence:.2f}
- overall: {proposal.overall_confidence:.2f}
- low confidence: {"yes" if proposal.low_confidence else "no"}

Canonical naming committed:
{"yes" if canonical_committed else "no"}

OCR working-image notes:
{chr(10).join(f"- {note}" for note in image_notes)}

Warnings:
{chr(10).join(f"- {warning}" for warning in proposal.warnings) if proposal.warnings else "- none"}

Next step:
- {next_step}
"""


def main() -> None:
    args = parse_args()
    wall_start = perf_counter()
    active_processing_ms = 0.0
    prompt_wait_ms = 0.0
    archive_dir = Path(args.archive_dir).resolve()
    paths = ensure_dirs(archive_dir)
    configure_tesseract(args.tesseract_cmd)

    source_path = Path(args.image).resolve() if args.image else choose_input_file(paths["incoming"])
    if not source_path.exists():
        raise FileNotFoundError(f"Input image not found: {source_path}")

    print("\nShows I Saw - Single Ticket Processor v1.0\n")
    print("This tool creates draft review artifacts only. It does not update tickets.js or publish to the live site.\n")
    print(f"Input: {source_path.name}")

    stage_start = perf_counter()
    original_image = load_image(source_path)
    image_load_ms = round((perf_counter() - stage_start) * 1000, 1)
    active_processing_ms += image_load_ms

    stage_start = perf_counter()
    cleaned_result = clean_ticket_image(original_image)
    cleanup_ms = round((perf_counter() - stage_start) * 1000, 1)
    active_processing_ms += cleanup_ms

    stage_start = perf_counter()
    ocr = run_ocr_variants(
        [
            ("display", cleaned_result.display_image),
            ("ocr-gray", cleaned_result.ocr_gray_image),
            ("ocr-binary", cleaned_result.ocr_binary_image),
        ],
        source_path=source_path,
    )
    ocr_ms = round((perf_counter() - stage_start) * 1000, 1)
    active_processing_ms += ocr_ms

    stage_start = perf_counter()
    proposal = propose_metadata(ocr, source_path=source_path)
    draft_ticket = build_draft_ticket(proposal)
    metadata_ms = round((perf_counter() - stage_start) * 1000, 1)
    active_processing_ms += metadata_ms

    print("\nProposed metadata\n")
    print(f"artist:      {proposal.artist or '[unknown]'}")
    print(f"venue:       {proposal.venue or '[unknown]'}")
    print(f"year:        {proposal.year or '[unknown]'}")
    print(f"exactDate:   {proposal.exact_date or '[unknown]'}")
    print(f"slug:        {proposal.slug}")
    print(f"confidence:  core={proposal.core_metadata_confidence:.2f}, overall={proposal.overall_confidence:.2f}, ocr={ocr.average_confidence:.2f}")
    print(f"approval:    {'eligible for canonical approval' if proposal.canonical_eligible else 'blocked from canonical approval'}")
    print(f"ocr mode:    passes={ocr.pass_count}, early-exit={'yes' if ocr.early_exit_triggered else 'no'}")
    print(f"ocr stop:    {ocr.early_exit_reason or 'planned passes completed'}")
    if proposal.approval_blockers:
        print("blocked by:  " + "; ".join(proposal.approval_blockers))

    if proposal.warnings:
        print("\nWarnings")
        for warning in proposal.warnings:
            print(f"- {warning}")

    canonical_committed = False
    publish_requested = False
    published_successfully = False
    publish_error = ""
    source_route_status = "left_in_place"
    source_moved_path = source_path
    canonical_original_path = ""
    site_image_output = ""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    review_slug = build_review_slug(source_path)

    if not proposal.canonical_eligible:
        case_dir = paths["needs_review"] / f"{timestamp}-{source_path.stem}"
        case_dir.mkdir(parents=True, exist_ok=True)
        original_output = case_dir / "original.jpg"
        final_image_output = case_dir / "final-image.jpg"
        ocr_working_output = case_dir / "ocr-working.jpg"
        json_output = case_dir / "draft.json"
        contact_sheet_output = case_dir / "contact-sheet.jpg"
        notes_output = case_dir / "notes.txt"
        ocr_text_output = case_dir / "ocr.txt"
        ocr_debug_output = case_dir / "ocr-debug.json"
        save_destination_reason = "ticket is blocked from canonical approval: " + "; ".join(proposal.approval_blockers)
        print("\nThis ticket does not currently qualify for canonical approval.")
    else:
        prompt_start = perf_counter()
        publish_requested = args.yes or confirm(
            "\nApprove and publish this ticket?"
        )
        prompt_wait_ms += round((perf_counter() - prompt_start) * 1000, 1)
        canonical_committed = publish_requested
        if publish_requested:
            proposed_original = paths["originals"] / f"{proposal.slug}-original.jpg"
            proposed_final_image = paths["original_copy"] / f"{proposal.slug}-original.jpg"
            proposed_ocr_working = paths["ocr_working"] / f"{proposal.slug}-ocr-working.jpg"
            proposed_json = paths["draft_json"] / f"{proposal.slug}.json"
            proposed_contact_sheet = paths["contact_sheets"] / f"{proposal.slug}-contact-sheet.jpg"
            proposed_notes = paths["notes"] / f"{proposal.slug}.txt"
            proposed_ocr_text = paths["ocr"] / f"{proposal.slug}-ocr.txt"
            proposed_ocr_debug = paths["ocr"] / f"{proposal.slug}-ocr-debug.json"

            if canonical_paths_available(
                proposed_original,
                proposed_final_image,
                proposed_ocr_working,
                proposed_json,
                proposed_contact_sheet,
                proposed_notes,
                proposed_ocr_text,
                proposed_ocr_debug,
            ):
                original_output = proposed_original
                final_image_output = proposed_final_image
                canonical_original_path = str(proposed_final_image)
                ocr_working_output = proposed_ocr_working
                json_output = proposed_json
                contact_sheet_output = proposed_contact_sheet
                notes_output = proposed_notes
                ocr_text_output = proposed_ocr_text
                ocr_debug_output = proposed_ocr_debug
                save_destination_reason = "high-confidence metadata was approved and canonical paths were available"
            else:
                overwrite_existing = True
                if overwrite_existing:
                    original_output = proposed_original
                    final_image_output = proposed_final_image
                    canonical_original_path = str(proposed_final_image)
                    ocr_working_output = proposed_ocr_working
                    json_output = proposed_json
                    contact_sheet_output = proposed_contact_sheet
                    notes_output = proposed_notes
                    ocr_text_output = proposed_ocr_text
                    ocr_debug_output = proposed_ocr_debug
                    save_destination_reason = "approved publish flow is overwriting existing canonical files in place"
                else:
                    canonical_committed = False
                    case_dir = paths["needs_review"] / f"{timestamp}-{source_path.stem}"
                    case_dir.mkdir(parents=True, exist_ok=True)
                    original_output = case_dir / "original.jpg"
                    final_image_output = case_dir / "final-image.jpg"
                    ocr_working_output = case_dir / "ocr-working.jpg"
                    json_output = case_dir / "draft.json"
                    contact_sheet_output = case_dir / "contact-sheet.jpg"
                    notes_output = case_dir / "notes.txt"
                    ocr_text_output = case_dir / "ocr.txt"
                    ocr_debug_output = case_dir / "ocr-debug.json"
                    proposal.warnings.append("Canonical output already exists for this slug, so files were routed to needs-review instead of overwriting.")
                    save_destination_reason = "approved canonical paths already existed and overwrite was declined, so the run was routed to needs-review"
        else:
            case_dir = paths["needs_review"] / f"{timestamp}-{source_path.stem}"
            case_dir.mkdir(parents=True, exist_ok=True)
            original_output = case_dir / "original.jpg"
            final_image_output = case_dir / "final-image.jpg"
            ocr_working_output = case_dir / "ocr-working.jpg"
            json_output = case_dir / "draft.json"
            contact_sheet_output = case_dir / "contact-sheet.jpg"
            notes_output = case_dir / "notes.txt"
            ocr_text_output = case_dir / "ocr.txt"
            ocr_debug_output = case_dir / "ocr-debug.json"
            save_destination_reason = "ticket was left in review-only mode because publish was not approved"
    stage_start = perf_counter()
    shutil.copy2(source_path, original_output)
    shutil.copy2(source_path, final_image_output)
    save_jpeg(cleaned_result.display_image, ocr_working_output)

    contact_sheet = build_contact_sheet(
        original=original_image,
        cleaned=cleaned_result.display_image,
        title=f"Shows I Saw Review: {proposal.slug or review_slug}",
    )
    save_jpeg(contact_sheet, contact_sheet_output)

    payload = build_review_payload(
        source_path=source_path,
        draft_ticket=draft_ticket,
        proposal=proposal,
        ocr=ocr,
        cleaned_display_path=str(ocr_working_output),
        original_copy_path=str(original_output),
        canonical_committed=canonical_committed,
    )
    if not canonical_committed:
        payload["reviewSlug"] = review_slug
    json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

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
                "coreMetadataConfidence": round(proposal.core_metadata_confidence, 3),
                "canonicalEligible": proposal.canonical_eligible,
                "approvalBlockers": proposal.approval_blockers,
                "ocrPassCount": ocr.pass_count,
                "ocrEarlyExitTriggered": ocr.early_exit_triggered,
                "ocrEarlyExitReason": ocr.early_exit_reason,
                "topArtistCandidates": proposal.artist_candidates,
                "rejectedArtistCandidates": proposal.rejected_artist_candidates,
                "topVenueCandidates": proposal.venue_candidates,
                "dateCandidates": proposal.date_candidates,
                "filenamePriorsUsed": proposal.filename_priors_used,
                "filenamePriors": proposal.filename_priors,
                "filenamePriorBreakdown": proposal.filename_priors,
                "multiArtistParsingAttempted": proposal.selector_debug.get("multiArtistParsingAttempted", False),
                "multiArtistParsingMatched": proposal.selector_debug.get("multiArtistParsingMatched", False),
                "venueTailSplittingAttempted": proposal.selector_debug.get("venueTailSplittingAttempted", False),
                "venueTailSplittingMatched": proposal.selector_debug.get("venueTailSplittingMatched", False),
                "activeParserStagesRun": proposal.selector_debug.get("parserStagesRun", []),
                "functionChain": proposal.selector_debug.get("functionChain", []),
                "finalSelectorWinner": proposal.selector_debug.get("finalSelectorWinner", ""),
                "finalSelectorReason": proposal.selector_debug.get("finalSelectorReason", ""),
                "selectorDebug": proposal.selector_debug,
                "chosenArtistExplanation": proposal.artist_candidates[0]["reasons"] if proposal.artist_candidates else [],
                "rawOcrTextByPass": [
                    {
                        "variant": run.get("variant", ""),
                        "config": run.get("config", ""),
                        "rawText": run.get("rawText", ""),
                        "cleanedText": run.get("cleanedText", ""),
                    }
                    for run in ocr.debug_runs
                ],
                "runs": ocr.debug_runs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    notes_text = build_notes_text(
        source_path=source_path,
        original_output=original_output,
        final_image_output=final_image_output,
        ocr_working_output=ocr_working_output,
        json_output=json_output,
        contact_sheet_output=contact_sheet_output,
        proposal=proposal,
        ocr=ocr,
        image_notes=cleaned_result.notes,
        canonical_committed=canonical_committed,
        review_slug=review_slug,
    )
    notes_output.write_text(notes_text, encoding="utf-8")
    save_ms = round((perf_counter() - stage_start) * 1000, 1)
    active_processing_ms += save_ms

    if canonical_committed and publish_requested:
        publish_plan, publish_error = prepare_site_write_plan(json_output, archive_dir=archive_dir)
        if publish_plan is not None:
            print_site_write_preview(publish_plan)
            write_site_write_plan(publish_plan)
            published_successfully = True
            site_image_output = str(publish_plan.final_site_image)
    if source_path.parent.resolve() == paths["incoming"].resolve():
        if published_successfully:
            source_moved_path = move_source_out_of_incoming(source_path, paths["incoming"], paths["processed"])
            source_route_status = "processed"
        else:
            source_moved_path = move_source_out_of_incoming(source_path, paths["incoming"], paths["rejected"])
            source_route_status = "rejected"

    wall_total_ms = round((perf_counter() - wall_start) * 1000, 1)

    print("\nSaved outputs\n")
    print(f"save mode:     {'canonical committed paths' if canonical_committed else 'review/needs-review'}")
    print(f"save reason:   {save_destination_reason}")
    print("final canonical image uses original (no enhancement applied)")
    if source_route_status == "processed":
        print(f"source file moved to: processed/ ({source_moved_path.name})")
    elif source_route_status == "rejected":
        print(f"source file moved to: rejected/ ({source_moved_path.name})")
    else:
        print("source file left in place: incoming/")
    print(f"canonical original stored at: {canonical_original_path or '[not written]'}")
    print(f"site image stored at: {site_image_output or '[not written]'}")
    print(f"source copy:    {original_output}")
    print(f"final image:   {final_image_output}")
    print(f"ocr working:   {ocr_working_output}")
    print(f"draft json:    {json_output}")
    print(f"ocr text:      {ocr_text_output}")
    print(f"ocr debug:     {ocr_debug_output}")
    print(f"contact sheet: {contact_sheet_output}")
    print(f"notes:         {notes_output}")
    print("\nTiming")
    print(f"image load:    {image_load_ms} ms")
    print(f"cleanup:       {cleanup_ms} ms")
    print(f"ocr:           {ocr_ms} ms")
    print(f"metadata:      {metadata_ms} ms")
    print(f"save:          {save_ms} ms")
    for index, run in enumerate(ocr.debug_runs, start=1):
        print(f"ocr pass {index}:   {run['variant']} {run['config']} {run['runtimeMs']} ms")
    print(f"prompt wait:   {round(prompt_wait_ms, 1)} ms")
    print(f"active total:  {round(active_processing_ms, 1)} ms")
    print(f"wall total:    {wall_total_ms} ms")
    print("\nStatus")
    print("canonical naming committed" if canonical_committed else "needs review before canonical naming")
    if published_successfully:
        print("\nTicket published successfully\n")
        print(f"Slug: {proposal.slug}")
        print("Image: canonical original-copy")
        print("tickets.js: updated")
        print("share page: created/updated")
    elif publish_requested and publish_error:
        print("\nPublish blocked\n")
        print(publish_error)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        sys.exit(1)
