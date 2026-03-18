from __future__ import annotations

from datetime import datetime
from pathlib import Path

from scan_splitting import (
    export_region_images,
    list_supported_images,
    load_image,
    reserve_output_path,
    save_image,
    split_ticket_scan,
)


def choose_source_file(candidates: list[Path], base_dir: Path) -> Path:
    if not candidates:
        raise FileNotFoundError("No supported images found in archive-ingest/incoming/.")

    print("\nAvailable multi-ticket scan sources\n")
    for index, path in enumerate(candidates, start=1):
        print(f"{index}. {path.relative_to(base_dir)}")

    while True:
        choice = input("\nChoose a file number: ").strip()
        if choice.isdigit():
            selected_index = int(choice)
            if 1 <= selected_index <= len(candidates):
                return candidates[selected_index - 1]
        print("Please enter a valid number from the list above.")


def main() -> None:
    script_path = Path(__file__).resolve()
    archive_ingest_dir = script_path.parent
    incoming_dir = archive_ingest_dir / "incoming"
    split_dir = archive_ingest_dir / "working" / "split"
    notes_dir = archive_ingest_dir / "review" / "notes"
    contact_sheet_dir = archive_ingest_dir / "review" / "contact-sheets"

    for folder in [split_dir, notes_dir, contact_sheet_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    print("\nShows I Saw - Multi-Ticket Scan Splitter v1.0\n")
    print("This tool creates review-only split candidates. It does not publish, modify the live site, or overwrite originals.\n")

    source_path = choose_source_file(list_supported_images(incoming_dir), archive_ingest_dir)
    print(f"\nSelected scan: {source_path.name}\n")
    print("Detecting ticket regions... please wait.\n")

    image = load_image(source_path)
    result = split_ticket_scan(image, source_path.name)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"{source_path.stem}-{timestamp}"

    split_paths = export_region_images(image, result.regions, split_dir, stem)
    preview_path = reserve_output_path(contact_sheet_dir / f"{stem}-preview.jpg")
    contact_sheet_path = reserve_output_path(contact_sheet_dir / f"{stem}-split-contact-sheet.jpg")
    notes_path = reserve_output_path(notes_dir / f"{stem}-split-review.txt")

    save_image(result.preview_image, preview_path)
    save_image(result.contact_sheet, contact_sheet_path)
    notes_path.write_text(result.notes_text, encoding="utf-8")

    print("Done.\n")
    print(f"Detected regions:    {len(result.regions)}")
    for index, split_path in enumerate(split_paths, start=1):
        print(f"Split {index:02d}:          {split_path}")
    print(f"Preview overlay:    {preview_path}")
    print(f"Contact sheet:      {contact_sheet_path}")
    print(f"Review notes:       {notes_path}")
    print("\nManual review is required before any split file goes into process_ticket_image.py.\n")


if __name__ == "__main__":
    main()
