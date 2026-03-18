from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ticket_enrichment_support import (
    apply_enrichment_plan,
    candidate_draft_paths,
    load_external_facts,
    load_writer_context,
    prepare_enrichment_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch enrich approval-eligible Shows I Saw ticket drafts."
    )
    parser.add_argument(
        "--archive-dir",
        default=str(Path(__file__).resolve().parent),
        help="Archive ingest root. Defaults to the folder containing this script.",
    )
    parser.add_argument(
        "--external-facts",
        default=None,
        help="Optional local JSON file containing factual enrichment records keyed by slug or filename stem.",
    )
    parser.add_argument(
        "--auto-ready",
        action="store_true",
        help="Apply enrichment automatically for tickets that become READY FOR APPROVAL.",
    )
    return parser.parse_args()


def confirm(prompt: str) -> bool:
    response = input(f"{prompt} [y/N]: ").strip().lower()
    return response in {"y", "yes"}


def print_plan_summary(plan) -> None:
    ticket = plan.enriched_payload.get("ticket", {})
    print("\n---\n")
    print(ticket.get("slug") or plan.draft_path.stem)
    print(f"status: {plan.status}")
    print(f"artist: {ticket.get('artist', '') or '[unknown]'}")
    print(f"venue:  {ticket.get('venue', '') or '[unknown]'}")
    print(f"year:   {ticket.get('year', '') or '[unknown]'}")
    print("fields filled: " + (", ".join(plan.fields_filled) or "none"))
    print("field sources:")
    for label, fields in plan.grouped_sources.items():
        print(f"  {label}: " + (", ".join(fields) or "none"))
    if plan.legacy_match.get("matched"):
        print(f"legacy match: {plan.legacy_match.get('sourcePath', '')} (score {plan.legacy_match.get('score', '')})")
        print("legacy importable: " + (", ".join(plan.legacy_match.get("importableFields", [])) or "none"))
    else:
        print("legacy match: none")
    top_candidates = plan.legacy_match.get("topCandidates", [])
    if top_candidates:
        print("top legacy candidates:")
        for candidate in top_candidates[:3]:
            print(
                "  "
                + f"{candidate.get('slug', '')} "
                + f"(score {candidate.get('score', '')}) "
                + f"- importable: {', '.join(candidate.get('importableFields', [])) or 'none'}"
            )
    if plan.external_match.get("matched"):
        print(f"external facts: {plan.external_match.get('key', '')}")
    else:
        print("external facts: none")
    print("remaining missing: " + (", ".join(plan.remaining_missing) or "none"))
    print(f"ready for approval: {'yes' if plan.ready_for_approval else 'no'}")


def print_queue(plans: list) -> None:
    print("\nEnrichment Queue\n")
    for plan in plans:
        ticket = plan.enriched_payload.get("ticket", {})
        print(ticket.get("slug") or plan.draft_path.stem)
        print(f"  {ticket.get('artist', '') or '[unknown artist]'} / {ticket.get('venue', '') or '[unknown venue]'} / {ticket.get('year', '') or '[unknown year]'}")
        print(f"  {plan.status}")
        if plan.remaining_missing:
            print(f"  missing: {', '.join(plan.remaining_missing)}")


def main() -> None:
    args = parse_args()
    archive_dir = Path(args.archive_dir).resolve()
    external_facts = load_external_facts(Path(args.external_facts).resolve()) if args.external_facts else {}
    writer_context = load_writer_context(archive_dir)
    draft_paths = candidate_draft_paths(archive_dir)

    if not draft_paths:
        print("\nNo approval-eligible or approved draft JSON files were found.\n")
        return

    plans = [
        prepare_enrichment_plan(
            draft_path,
            archive_dir=archive_dir,
            writer_context=writer_context,
            external_facts=external_facts,
        )
        for draft_path in draft_paths
    ]

    print_queue(plans)

    updated = 0
    skipped = 0
    ready = 0

    for plan in plans:
        print_plan_summary(plan)
        should_apply = False
        if args.auto_ready and plan.ready_for_approval:
            should_apply = True
            print("auto-ready: enrichment applied")
        else:
            should_apply = confirm("apply enrichment to this draft?")

        if not should_apply:
            skipped += 1
            continue

        apply_enrichment_plan(plan)
        updated += 1
        if plan.ready_for_approval:
            ready += 1

    print("\nSummary\n")
    print(f"updated: {updated}")
    print(f"ready:   {ready}")
    print(f"skipped: {skipped}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        sys.exit(1)
