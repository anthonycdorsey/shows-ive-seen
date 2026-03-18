from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from dataclasses import dataclass

from site_writer_support import (
    build_site_ticket,
    build_ticket_object_js,
    build_tickets_js_preview,
    build_venue_context,
    choose_approved_draft,
    load_venue_lookup,
    parse_existing_ticket_blocks,
    render_share_page,
    update_tickets_js,
    validate_site_ticket,
)

COMPAT_APPROVAL_FIELDS = [
    "artist",
    "venue",
    "year",
    "city",
    "state",
    "country",
    "slug",
    "img",
]


@dataclass
class SiteWritePlan:
    draft_path: Path
    tickets_js_path: Path
    share_page_path: Path
    final_site_image: Path
    final_image_source: Path
    updated_tickets_js: str
    share_page_html: str
    site_ticket: dict
    change_type: str
    changed_fields: list[str]
    normalized_legacy_image: bool
    image_resolution_reason: str
    completion_debug: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview and write an approved ingest draft into tickets.js and the share page structure."
    )
    parser.add_argument("draft_json", nargs="?", help="Path to an approved draft JSON file.")
    parser.add_argument(
        "--repair-image",
        action="store_true",
        help="If a legacy approved original is found outside the current canonical location, copy it into originals/original-copy before continuing.",
    )
    return parser.parse_args()


def confirm(prompt: str) -> bool:
    response = input(f"{prompt} [y/N]: ").strip().lower()
    return response in {"y", "yes"}


def resolve_payload_path(raw_value: str, base_dir: Path) -> Path | None:
    if not raw_value:
        return None
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def normalize_draft_payload_for_site(draft_payload: dict, archive_dir: Path) -> tuple[dict, dict]:
    normalized = dict(draft_payload)
    normalized["ticket"] = dict(draft_payload.get("ticket", {}))
    source = dict(draft_payload.get("source", {}))

    if source.get("selectedIncomingFile") and not source.get("inputFile"):
        source["inputFile"] = source["selectedIncomingFile"]

    source_original = resolve_payload_path(source.get("sourceOriginal", ""), archive_dir.parent)
    if source_original and not source.get("savedOriginal"):
        source["savedOriginal"] = str(source_original)

    saved_original = resolve_payload_path(source.get("savedOriginal", ""), archive_dir.parent)
    if saved_original:
        source["savedOriginal"] = str(saved_original)

    normalized["source"] = source

    compat_missing = [
        field for field in COMPAT_APPROVAL_FIELDS
        if not normalized["ticket"].get(field)
    ]
    manual_compat_approved = not compat_missing
    approval_info = {
        "canonicalNamingCommitted": bool(draft_payload.get("canonicalNamingCommitted")),
        "manualCompatApproved": manual_compat_approved,
        "compatMissingFields": compat_missing,
    }
    return normalized, approval_info


def resolve_final_image_source(archive_dir: Path, slug: str, draft_payload: dict) -> tuple[Path | None, list[Path], str]:
    checked_paths: list[Path] = []
    canonical_original = archive_dir / "originals" / "original-copy" / f"{slug}-original.jpg"
    checked_paths.append(canonical_original)
    if canonical_original.exists():
        return canonical_original, checked_paths, "found in current canonical original-copy location"

    legacy_original = archive_dir / "originals" / f"{slug}-original.jpg"
    checked_paths.append(legacy_original)
    if legacy_original.exists():
        return legacy_original, checked_paths, "found in older originals/ legacy location"

    saved_original = draft_payload.get("source", {}).get("savedOriginal", "")
    if saved_original:
        payload_path = Path(saved_original)
        checked_paths.append(payload_path)
        if payload_path.exists():
            return payload_path, checked_paths, "found via savedOriginal path in draft payload"

    source_original = draft_payload.get("source", {}).get("sourceOriginal", "")
    if source_original:
        payload_path = resolve_payload_path(source_original, archive_dir.parent)
        if payload_path is not None:
            checked_paths.append(payload_path)
            if payload_path.exists():
                return payload_path, checked_paths, "found via sourceOriginal path in compatibility draft payload"

    input_file_stem = Path(draft_payload.get("source", {}).get("inputFile", "")).stem.lower()
    candidate_dirs = [
        archive_dir / "review" / "needs-review",
        archive_dir / "originals",
    ]
    patterns = [
        f"{slug}-original.jpg",
        f"{slug}.jpg",
        "original.jpg",
        "final-image.jpg",
    ]
    for directory in candidate_dirs:
        if not directory.exists():
            continue
        for pattern in patterns:
            for path in directory.rglob(pattern):
                checked_paths.append(path)
                parent_name = path.parent.name.lower()
                slug_match = slug.lower() in path.as_posix().lower() or slug.lower() in parent_name
                input_match = input_file_stem and input_file_stem in parent_name
                if pattern in {"original.jpg", "final-image.jpg"}:
                    if slug_match or input_match:
                        return path, checked_paths, "found via legacy review/originals fallback search"
                elif slug_match:
                    return path, checked_paths, "found via legacy review/originals fallback search"

    return None, checked_paths, "approved original image was not found in canonical, legacy, payload, or review fallback locations"


def print_missing_image_diagnostics(slug: str, draft_payload: dict, checked_paths: list[Path], reason: str) -> None:
    print("\nApproved original image is missing\n")
    print(f"slug: {slug}")
    print(f"draft savedOriginal present: {'yes' if draft_payload.get('source', {}).get('savedOriginal') else 'no'}")
    if draft_payload.get("source", {}).get("savedOriginal"):
        print(f"draft savedOriginal value: {draft_payload['source']['savedOriginal']}")
    print("\nPaths checked")
    for path in checked_paths:
        print(path)
    print("\nWhy this likely happened")
    print(reason)
    print("\nNext steps")
    print(f"Re-run this ticket through the ingestion pipeline, or manually place the approved original in archive-ingest/originals/original-copy/{slug}-original.jpg.")


def prepare_site_write_plan(
    draft_path: Path,
    *,
    archive_dir: Path | None = None,
) -> tuple[SiteWritePlan | None, str | None]:
    archive_dir = archive_dir or Path(__file__).resolve().parent
    project_root = archive_dir.parent
    tickets_js_path = project_root / "tickets.js"
    share_root = project_root / "share"
    venue_lookup_path = archive_dir / "venue_lookup.json"

    draft_payload = json.loads(draft_path.read_text(encoding="utf-8"))
    draft_payload, approval_info = normalize_draft_payload_for_site(draft_payload, archive_dir)
    if not approval_info["canonicalNamingCommitted"] and not approval_info["manualCompatApproved"]:
        missing = ", ".join(approval_info["compatMissingFields"]) or "none"
        return None, (
            "This draft is not approved for canonical naming yet.\n"
            f"Compatibility approval blocked by missing fields: {missing}"
        )

    tickets_js_text = tickets_js_path.read_text(encoding="utf-8")
    existing_blocks = parse_existing_ticket_blocks(tickets_js_text)
    venue_context = build_venue_context(load_venue_lookup(venue_lookup_path), existing_blocks)

    site_ticket, completion_debug = build_site_ticket(
        draft_payload,
        venue_context,
        archive_dir,
        draft_path,
    )
    missing_fields = validate_site_ticket(site_ticket)
    if missing_fields:
        return None, "Missing fields: " + ", ".join(missing_fields)

    slug = site_ticket["slug"]
    final_image_source, checked_paths, image_resolution_reason = resolve_final_image_source(archive_dir, slug, draft_payload)
    canonical_image_target = archive_dir / "originals" / "original-copy" / f"{slug}-original.jpg"
    if final_image_source is None:
        diagnostics = [
            "Approved original image is missing.",
            f"slug: {slug}",
            f"draft savedOriginal present: {'yes' if draft_payload.get('source', {}).get('savedOriginal') else 'no'}",
            "paths checked:",
            *[str(path) for path in checked_paths],
            f"reason: {image_resolution_reason}",
        ]
        return None, "\n".join(diagnostics)

    normalized_legacy_image = False
    if final_image_source != canonical_image_target:
        canonical_image_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final_image_source, canonical_image_target)
        final_image_source = canonical_image_target
        normalized_legacy_image = True
        image_resolution_reason = "legacy approved original was copied into the current canonical original-copy location"

    final_site_image = project_root / site_ticket["img"]
    share_page_path = share_root / slug / "index.html"
    existing_ticket = next((block.fields for block in existing_blocks if block.slug == slug), None)
    object_preview, changed_fields = build_tickets_js_preview(existing_ticket, site_ticket)
    updated_tickets_js, change_type = update_tickets_js(tickets_js_text, slug, object_preview)
    share_page_html = render_share_page(site_ticket)

    return SiteWritePlan(
        draft_path=draft_path,
        tickets_js_path=tickets_js_path,
        share_page_path=share_page_path,
        final_site_image=final_site_image,
        final_image_source=final_image_source,
        updated_tickets_js=updated_tickets_js,
        share_page_html=share_page_html,
        site_ticket=site_ticket,
        change_type=change_type,
        changed_fields=changed_fields,
        normalized_legacy_image=normalized_legacy_image,
        image_resolution_reason=image_resolution_reason,
        completion_debug=completion_debug,
    ), None


def write_site_write_plan(plan: SiteWritePlan) -> None:
    plan.tickets_js_path.write_text(plan.updated_tickets_js, encoding="utf-8")
    plan.share_page_path.parent.mkdir(parents=True, exist_ok=True)
    plan.share_page_path.write_text(plan.share_page_html, encoding="utf-8")
    shutil.copy2(plan.final_image_source, plan.final_site_image)


def print_site_write_preview(plan: SiteWritePlan) -> None:
    print("\nProposed Ticket Object\n")
    print(build_ticket_object_js(plan.site_ticket))

    print("\nChange Type")
    print(plan.change_type)

    print("\nRequired-Field Validation")
    print("PASS")

    print("\nField Sources")
    print("approved metadata: " + (", ".join(plan.completion_debug.get("sources", {}).get("approvedMetadata", [])) or "none"))
    print("venue lookup:      " + (", ".join(plan.completion_debug.get("sources", {}).get("venueLookup", [])) or "none"))
    print("legacy curated:    " + (", ".join(plan.completion_debug.get("sources", {}).get("legacyCurated", [])) or "none"))
    print("auto drafted:      " + (", ".join(plan.completion_debug.get("sources", {}).get("autoDrafted", [])) or "none"))
    print("still missing:     " + (", ".join(plan.completion_debug.get("sources", {}).get("missing", [])) or "none"))

    legacy_match = plan.completion_debug.get("legacyMatch", {})
    if legacy_match.get("matched"):
        print("\nLegacy Match")
        print(f"path: {legacy_match.get('sourcePath', '')}")
        print(f"score: {legacy_match.get('score', '')}")
    else:
        print("\nLegacy Match")
        print(legacy_match.get("reason", "none"))

    print("\nFiles To Change")
    print(plan.tickets_js_path)
    print(plan.share_page_path)
    print(plan.final_site_image)

    print("\nFinal Image Source")
    print(plan.final_image_source)
    print(f"resolution: {plan.image_resolution_reason}")
    if plan.normalized_legacy_image:
        print("Normalized legacy image to canonical originals folder")


def main() -> None:
    args = parse_args()
    archive_dir = Path(__file__).resolve().parent
    draft_dir = archive_dir / "review" / "draft-json"

    draft_path = Path(args.draft_json).resolve() if args.draft_json else choose_approved_draft(draft_dir)
    plan, error = prepare_site_write_plan(draft_path, archive_dir=archive_dir)
    if error:
        print("\nRequired-field validation failed\n")
        print(error)
        print("\nNo files were written.")
        return

    print_site_write_preview(plan)

    if plan.change_type == "UPDATE":
        print("\nChanged Fields")
        print(", ".join(plan.changed_fields) if plan.changed_fields else "No field-level changes detected.")

    if not confirm("\nWrite these site updates"):
        print("\nNo files were written.")
        return

    write_site_write_plan(plan)

    print("\nWrote site updates\n")
    print(f"tickets.js:    {plan.tickets_js_path}")
    print(f"share page:    {plan.share_page_path}")
    print(f"site image:    {plan.final_site_image}")
    print(f"final source:  {plan.final_image_source}")
    if plan.normalized_legacy_image:
        print("Normalized legacy image to canonical originals folder")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        sys.exit(1)
