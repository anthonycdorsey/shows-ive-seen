from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from image_processing import (
    ContactSheetInputs,
    build_contact_sheet,
    build_processing_notes,
    load_image as load_ticket_image,
    process_ticket_image,
    reserve_output_path as reserve_image_output_path,
    save_image as save_ticket_image,
)
from ingest_ticket import create_review_artifacts_for_file
from scan_splitting import (
    export_region_images,
    list_supported_images,
    load_image as load_scan_image,
    reserve_output_path as reserve_split_output_path,
    save_image as save_split_image,
    split_ticket_scan,
)


SITE_BASE_URL = "https://www.anthonycdorsey.com/shows-ive-seen"


def to_review_relative(path: Path, project_root: Path) -> str:
    return str(path.relative_to(project_root)).replace("\\", "/")


def ensure_batch_directories(archive_ingest_dir: Path) -> dict[str, Path]:
    directories = {
        "incoming": archive_ingest_dir / "incoming",
        "split": archive_ingest_dir / "working" / "split",
        "cropped": archive_ingest_dir / "working" / "cropped",
        "enhanced": archive_ingest_dir / "working" / "enhanced",
        "contact_sheets": archive_ingest_dir / "review" / "contact-sheets",
        "notes": archive_ingest_dir / "review" / "notes",
        "manifests": archive_ingest_dir / "review" / "manifests",
        "batch_logs": archive_ingest_dir / "review" / "batch-logs",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def build_processing_output_paths(archive_ingest_dir: Path, source_path: Path, batch_stamp: str) -> dict[str, Path]:
    base_name = f"{source_path.stem}-{batch_stamp}"
    return {
        "archive": reserve_image_output_path(archive_ingest_dir / "working" / "cropped" / f"{base_name}-archive.jpg"),
        "share": reserve_image_output_path(archive_ingest_dir / "working" / "enhanced" / f"{base_name}-share.jpg"),
        "preview": reserve_image_output_path(archive_ingest_dir / "working" / "enhanced" / f"{base_name}-preview.jpg"),
        "contact_sheet": reserve_image_output_path(archive_ingest_dir / "review" / "contact-sheets" / f"{base_name}-contact-sheet.jpg"),
        "notes": reserve_image_output_path(archive_ingest_dir / "review" / "notes" / f"{base_name}-image-review.txt"),
    }


def write_manifest(manifests_dir: Path, stem: str, payload: dict) -> Path:
    manifest_path = reserve_image_output_path(manifests_dir / f"{stem}.json")
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def process_split_candidate(
    split_path: Path,
    archive_ingest_dir: Path,
    project_root: Path,
    batch_stamp: str,
) -> dict:
    source_image = load_ticket_image(split_path)
    result = process_ticket_image(source_image)
    output_paths = build_processing_output_paths(archive_ingest_dir, split_path, batch_stamp)

    save_ticket_image(result.archive_image, output_paths["archive"])
    save_ticket_image(result.share_image, output_paths["share"])
    save_ticket_image(result.preview_image, output_paths["preview"])

    contact_sheet = build_contact_sheet(
        ContactSheetInputs(
            original=source_image,
            archive_candidate=result.archive_image,
            share_candidate=result.share_image,
            preview_candidate=result.preview_image,
            title="Shows I Saw - Image Processing Review",
            subtitle=f"Source: {split_path.name}",
        )
    )
    save_ticket_image(contact_sheet, output_paths["contact_sheet"])

    notes_text = build_processing_notes(
        source_path=split_path,
        archive_output=output_paths["archive"],
        share_output=output_paths["share"],
        preview_output=output_paths["preview"],
        contact_sheet_output=output_paths["contact_sheet"],
        result=result,
    )
    output_paths["notes"].write_text(notes_text, encoding="utf-8")

    ambiguity = []
    if result.detection.fallback_used:
        ambiguity.append("Image processing used fallback ticket-bound detection.")
    if result.detection.confidence < 0.20:
        ambiguity.append(f"Image processing confidence is low ({result.detection.confidence:.2f}).")

    return {
        "processed_archive_candidate": to_review_relative(output_paths["archive"], project_root),
        "processed_share_candidate": to_review_relative(output_paths["share"], project_root),
        "processed_archive_candidate_path": str(output_paths["archive"]),
        "image_preview": to_review_relative(output_paths["preview"], project_root),
        "contact_sheet": to_review_relative(output_paths["contact_sheet"], project_root),
        "processing_notes": to_review_relative(output_paths["notes"], project_root),
        "image_processing": {
            "detection_method": result.detection.method,
            "detection_confidence": round(result.detection.confidence, 3),
            "fallback_used": result.detection.fallback_used,
            "rotation_applied": round(result.rotation_applied, 3),
            "rotation_reason": result.rotation_reason,
            "crop_padding": list(result.crop_padding),
            "enhancement_summary": result.enhancement_summary,
            "framing_summary": result.framing_summary,
            "share_summary": result.share_summary,
        },
        "ambiguity": ambiguity,
    }


def build_batch_summary(
    batch_started_at: str,
    batch_finished_at: str,
    processed_scans: list[dict],
    failures: list[dict],
) -> dict:
    split_candidates = sum(len(scan["tickets"]) for scan in processed_scans)
    metadata_drafts = sum(
        1
        for scan in processed_scans
        for ticket in scan["tickets"]
        if ticket.get("metadata_draft_json")
    )
    ambiguous_results = [
        {
            "source_scan": scan["source_scan"],
            "ticket_candidate": ticket.get("split_candidate"),
            "issues": ticket.get("ambiguity", []),
        }
        for scan in processed_scans
        for ticket in scan["tickets"]
        if ticket.get("ambiguity")
    ]
    ambiguous_results.extend(
        {
            "source_scan": scan["source_scan"],
            "ticket_candidate": "",
            "issues": scan.get("ambiguity", []),
        }
        for scan in processed_scans
        if scan.get("ambiguity")
    )

    return {
        "reviewStatus": "pending_human_review",
        "generated_at": batch_finished_at,
        "scans_processed": len(processed_scans),
        "split_candidates_generated": split_candidates,
        "ticket_candidates_processed": split_candidates,
        "metadata_drafts_generated": metadata_drafts,
        "ambiguous_results": ambiguous_results,
        "batchStartedAt": batch_started_at,
        "batchFinishedAt": batch_finished_at,
        "sourceScanCount": len(processed_scans),
        "splitCandidateCount": split_candidates,
        "ticketCandidateCount": split_candidates,
        "metadataDraftCount": metadata_drafts,
        "failureCount": len(failures),
        "failures": failures,
        "ambiguousResults": ambiguous_results,
        "protections": [
            "Never appends to tickets.js automatically.",
            "Never copies processed images into the live site automatically.",
            "Never creates live share folders automatically.",
            "Never modifies website HTML, CSS, or JavaScript.",
        ],
        "processedScans": processed_scans,
    }


def main() -> None:
    script_path = Path(__file__).resolve()
    archive_ingest_dir = script_path.parent
    project_root = archive_ingest_dir.parent
    directories = ensure_batch_directories(archive_ingest_dir)

    batch_started = datetime.now()
    batch_stamp = batch_started.strftime("%Y%m%d-%H%M%S")

    print("\nShows I Saw - Review-Only Batch Ingest Controller v1.0\n")
    print("This tool orchestrates split, image processing, and metadata draft generation.")
    print("It never updates tickets.js, never copies files into the live site, and never publishes automatically.\n")

    incoming_files = list_supported_images(directories["incoming"])
    if not incoming_files:
        print("No supported scan images were found in archive-ingest/incoming/.")
        return

    processed_scans: list[dict] = []
    failures: list[dict] = []

    for scan_path in incoming_files:
        print(f"Processing scan: {scan_path.name}")
        scan_record = {
            "source_scan": to_review_relative(scan_path, project_root),
            "split_review_preview": "",
            "split_contact_sheet": "",
            "split_notes": "",
            "ambiguity": [],
            "warnings": [],
            "failures": [],
            "tickets": [],
        }

        try:
            scan_image = load_scan_image(scan_path)
            split_result = split_ticket_scan(scan_image, scan_path.name)
            split_stem = f"{scan_path.stem}-{batch_stamp}"

            split_paths = export_region_images(scan_image, split_result.regions, directories["split"], split_stem)
            preview_path = reserve_split_output_path(directories["contact_sheets"] / f"{split_stem}-preview.jpg")
            contact_sheet_path = reserve_split_output_path(directories["contact_sheets"] / f"{split_stem}-split-contact-sheet.jpg")
            split_notes_path = reserve_split_output_path(directories["notes"] / f"{split_stem}-split-review.txt")

            save_split_image(split_result.preview_image, preview_path)
            save_split_image(split_result.contact_sheet, contact_sheet_path)
            split_notes_path.write_text(split_result.notes_text, encoding="utf-8")

            scan_record["split_review_preview"] = to_review_relative(preview_path, project_root)
            scan_record["split_contact_sheet"] = to_review_relative(contact_sheet_path, project_root)
            scan_record["split_notes"] = to_review_relative(split_notes_path, project_root)
            if len(split_result.regions) == 1:
                scan_record["ambiguity"].append("Splitter returned one broad region. Confirm no additional tickets were missed.")
                scan_record["warnings"].append("Splitter returned one broad region. Confirm no additional tickets were missed.")

            if not split_paths:
                failure_record = {
                    "source_scan": scan_record["source_scan"],
                    "stage": "split",
                    "message": "No split candidates were generated from this scan.",
                }
                scan_record["failures"].append(failure_record)
                failures.append({
                    **failure_record,
                })
                processed_scans.append(scan_record)
                print("  No split candidates were generated.\n")
                continue

            for split_index, split_path in enumerate(split_paths, start=1):
                ticket_record = {
                    "source_scan": scan_record["source_scan"],
                    "split_candidate": to_review_relative(split_path, project_root),
                    "split_index": split_index,
                    "processed_archive_candidate": "",
                    "processed_share_candidate": "",
                    "metadata_draft_json": "",
                    "metadata_notes": "",
                    "share_page_draft": "",
                    "ticket_object_block": "",
                    "processing_notes": "",
                    "contact_sheet": "",
                    "warnings": [],
                    "failures": [],
                    "status": "pending_human_review",
                    "ambiguity": [],
                }

                try:
                    processing_record = process_split_candidate(split_path, archive_ingest_dir, project_root, batch_stamp)
                    ticket_record.update(processing_record)
                    ticket_record["warnings"].extend(processing_record.get("ambiguity", []))

                    metadata_result = create_review_artifacts_for_file(
                        Path(ticket_record["processed_archive_candidate_path"]),
                        archive_ingest_dir,
                        project_root,
                        SITE_BASE_URL,
                    )

                    ticket_record["metadata_draft_json"] = to_review_relative(metadata_result.draft_json_path, project_root)
                    ticket_record["metadata_notes"] = to_review_relative(metadata_result.notes_path, project_root)
                    ticket_record["share_page_draft"] = to_review_relative(metadata_result.share_page_path, project_root)
                    ticket_record["ticket_object_block"] = metadata_result.ticket_object
                    ticket_record["metadata_validation_warnings"] = metadata_result.validation_warnings
                    if metadata_result.validation_warnings:
                        ticket_record["ambiguity"].extend(metadata_result.validation_warnings)
                        ticket_record["warnings"].extend(metadata_result.validation_warnings)
                    ticket_record["ticket_slug"] = metadata_result.ticket["slug"]
                    ticket_record.pop("processed_archive_candidate_path", None)
                except Exception as exc:  # noqa: BLE001
                    ticket_record["status"] = "ticket_pipeline_failed"
                    ticket_record["ambiguity"].append(str(exc))
                    ticket_record["warnings"].append(str(exc))
                    ticket_record.pop("processed_archive_candidate_path", None)
                    failure_record = {
                        "source_scan": scan_record["source_scan"],
                        "split_candidate": ticket_record["split_candidate"],
                        "stage": "ticket_pipeline",
                        "message": str(exc),
                    }
                    ticket_record["failures"].append(failure_record)
                    failures.append(failure_record)

                manifest_stem = f"{split_path.stem}-manifest"
                manifest_path = write_manifest(directories["manifests"], manifest_stem, ticket_record)
                ticket_record["manifest"] = to_review_relative(manifest_path, project_root)
                scan_record["tickets"].append(ticket_record)

            processed_scans.append(scan_record)
            print(f"  Split candidates: {len(scan_record['tickets'])}\n")
        except Exception as exc:  # noqa: BLE001
            failure_record = {
                "source_scan": scan_record["source_scan"],
                "stage": "scan_pipeline",
                "message": str(exc),
            }
            scan_record["failures"].append(failure_record)
            scan_record["warnings"].append(str(exc))
            failures.append(failure_record)
            processed_scans.append(scan_record)
            print(f"  Failed: {exc}\n")

    batch_finished = datetime.now()
    summary_payload = build_batch_summary(
        batch_started.isoformat(timespec="seconds"),
        batch_finished.isoformat(timespec="seconds"),
        processed_scans,
        failures,
    )
    summary_path = reserve_image_output_path(directories["batch_logs"] / f"batch-{batch_stamp}.json")
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Batch complete.\n")
    print(f"Scans processed:            {summary_payload['sourceScanCount']}")
    print(f"Split candidates created:   {summary_payload['splitCandidateCount']}")
    print(f"Ticket candidates handled:  {summary_payload['ticketCandidateCount']}")
    print(f"Metadata drafts generated:  {summary_payload['metadataDraftCount']}")
    print(f"Failures:                   {summary_payload['failureCount']}")
    print(f"Batch summary:              {summary_path}")
    print("\nManual review is still required before anything is published or copied into the live site.\n")


if __name__ == "__main__":
    main()
