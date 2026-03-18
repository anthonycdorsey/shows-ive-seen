from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

from approved_ticket_to_site import resolve_final_image_source


def main() -> None:
    archive_dir = Path(__file__).resolve().parent
    draft_dir = archive_dir / "review" / "draft-json"
    canonical_dir = archive_dir / "originals" / "original-copy"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    normalized_count = 0
    skipped_count = 0
    missing_count = 0

    for draft_path in sorted(draft_dir.glob("*.json")):
        try:
            payload = json.loads(draft_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            skipped_count += 1
            continue

        if not payload.get("canonicalNamingCommitted"):
            skipped_count += 1
            continue

        slug = payload.get("ticket", {}).get("slug", "")
        if not slug:
            skipped_count += 1
            continue

        resolved_path, checked_paths, _ = resolve_final_image_source(archive_dir, slug, payload)
        canonical_target = canonical_dir / f"{slug}-original.jpg"

        if resolved_path is None:
            missing_count += 1
            print(f"MISSING: {slug}")
            continue

        if resolved_path.resolve() == canonical_target.resolve():
            skipped_count += 1
            continue

        shutil.copy2(resolved_path, canonical_target)
        normalized_count += 1
        print(f"NORMALIZED: {slug}")

    print("\nSummary")
    print(f"normalized: {normalized_count}")
    print(f"already canonical or skipped: {skipped_count}")
    print(f"missing source: {missing_count}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        sys.exit(1)
