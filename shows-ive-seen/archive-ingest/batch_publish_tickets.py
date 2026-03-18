from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

from approved_ticket_to_site import prepare_site_write_plan, write_site_write_plan


@dataclass
class QueueItem:
    draft_path: Path
    slug: str
    artist: str
    venue: str
    year: str
    status: str
    blocker_summary: str
    plan: object | None
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch review and publish approved Shows I Saw ticket drafts."
    )
    parser.add_argument(
        "--archive-dir",
        default=str(Path(__file__).resolve().parent),
        help="Archive ingest root. Defaults to the folder containing this script.",
    )
    parser.add_argument(
        "--auto-ready",
        action="store_true",
        help="Auto-publish tickets that are fully READY without prompting.",
    )
    return parser.parse_args()


def confirm(prompt: str) -> bool:
    response = input(f"{prompt} [y/N]: ").strip().lower()
    return response in {"y", "yes"}


def load_approved_drafts(draft_dir: Path) -> list[Path]:
    approved: list[Path] = []
    for path in sorted(draft_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("canonicalNamingCommitted"):
            approved.append(path)
    return approved


def read_draft_summary(draft_path: Path) -> tuple[str, str, str, str]:
    payload = json.loads(draft_path.read_text(encoding="utf-8"))
    ticket = payload.get("ticket", {})
    return (
        ticket.get("slug", "") or payload.get("reviewSlug", "") or draft_path.stem,
        ticket.get("artist", ""),
        ticket.get("venue", ""),
        ticket.get("year", ""),
    )


def classify_plan(plan) -> str:
    sources = plan.completion_debug.get("sources", {})
    venue_lookup_used = bool(sources.get("venueLookup"))
    legacy_used = bool(sources.get("legacyCurated"))
    if venue_lookup_used or legacy_used:
        return "READY WITH LEGACY MERGE"
    return "READY"


def summarize_error(error: str | None) -> str:
    if not error:
        return ""
    first_line = next((line.strip() for line in error.splitlines() if line.strip()), "")
    if first_line.startswith("Missing fields:"):
        return first_line.replace("Missing fields:", "").strip()
    return first_line


def build_queue_items(approved_drafts: list[Path], archive_dir: Path) -> list[QueueItem]:
    items: list[QueueItem] = []
    for draft_path in approved_drafts:
        slug, artist, venue, year = read_draft_summary(draft_path)
        plan, error = prepare_site_write_plan(draft_path, archive_dir=archive_dir)
        if plan is None:
            items.append(
                QueueItem(
                    draft_path=draft_path,
                    slug=slug,
                    artist=artist,
                    venue=venue,
                    year=year,
                    status="BLOCKED",
                    blocker_summary=summarize_error(error),
                    plan=None,
                    error=error,
                )
            )
            continue

        items.append(
            QueueItem(
                draft_path=draft_path,
                slug=plan.site_ticket.get("slug", slug),
                artist=plan.site_ticket.get("artist", artist),
                venue=plan.site_ticket.get("venue", venue),
                year=plan.site_ticket.get("year", year),
                status=classify_plan(plan),
                blocker_summary="",
                plan=plan,
                error=None,
            )
        )
    return items


def print_queue_summary(items: list[QueueItem]) -> None:
    print("\nBatch Queue\n")
    for item in items:
        print(f"{item.slug or '[no-slug]'}")
        print(f"  {item.artist or '[unknown artist]'} / {item.venue or '[unknown venue]'} / {item.year or '[unknown year]'}")
        print(f"  {item.status}")
        if item.blocker_summary:
            print(f"  blocker: {item.blocker_summary}")


def print_item_preview(item: QueueItem) -> None:
    plan = item.plan
    if plan is None:
        return
    sources = plan.completion_debug.get("sources", {})
    print("\n---\n")
    print(f"{item.slug}")
    print(f"artist: {item.artist}")
    print(f"venue:  {item.venue}")
    print(f"year:   {item.year}")
    print(f"status: {item.status}")
    print(f"change: {plan.change_type}")
    print("field sources:")
    print("  OCR/approved:   " + (", ".join(sources.get("approvedMetadata", [])) or "none"))
    print("  venue lookup:   " + (", ".join(sources.get("venueLookup", [])) or "none"))
    print("  legacy curated: " + (", ".join(sources.get("legacyCurated", [])) or "none"))
    print("  auto drafted:   " + (", ".join(sources.get("autoDrafted", [])) or "none"))
    print("  still missing:  " + (", ".join(sources.get("missing", [])) or "none"))


def main() -> None:
    args = parse_args()
    archive_dir = Path(args.archive_dir).resolve()
    draft_dir = archive_dir / "review" / "draft-json"
    approved_drafts = load_approved_drafts(draft_dir)
    if not approved_drafts:
        print("\nNo approved draft JSON files were found.\n")
        return

    items = build_queue_items(approved_drafts, archive_dir)
    print_queue_summary(items)

    published = 0
    skipped = 0
    blocked = 0

    for item in items:
        if item.status == "BLOCKED":
            blocked += 1
            continue

        print_item_preview(item)
        should_publish = False
        if args.auto_ready and item.status == "READY":
            should_publish = True
            print("auto-ready: publishing without prompt")
        else:
            should_publish = confirm("publish this ticket?")

        if not should_publish:
            skipped += 1
            continue

        write_site_write_plan(item.plan)
        published += 1
        print("published")

    print("\nSummary\n")
    print(f"published: {published}")
    print(f"skipped:   {skipped}")
    print(f"blocked:   {blocked}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        sys.exit(1)
