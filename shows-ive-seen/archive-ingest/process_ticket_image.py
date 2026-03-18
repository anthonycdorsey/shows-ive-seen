from __future__ import annotations

from pathlib import Path
from datetime import datetime

from image_processing import (
    ContactSheetInputs,
    build_contact_sheet,
    build_processing_notes,
    list_supported_images,
    load_image,
    process_ticket_image,
    reserve_output_path,
    save_image,
)


def choose_source_file(candidates: list[Path]) -> Path:
    if not candidates:
        raise FileNotFoundError("No supported source images found in incoming/, originals/, or working/.")

    print("\nAvailable image sources\n")
    for index, path in enumerate(candidates, start=1):
        print(f"{index}. {path.relative_to(path.parents[2])}")

    while True:
        choice = input("\nChoose a file number: ").strip()
        if choice.isdigit():
            selected_index = int(choice)
            if 1 <= selected_index <= len(candidates):
                return candidates[selected_index - 1]
        print("Please enter a valid number from the list above.")


def gather_source_images(archive_ingest_dir: Path) -> list[Path]:
    candidate_dirs = [
        archive_ingest_dir / "incoming",
        archive_ingest_dir / "originals",
        archive_ingest_dir / "working" / "cropped",
        archive_ingest_dir / "working" / "enhanced",
    ]

    paths: list[Path] = []
    for folder in candidate_dirs:
        if not folder.exists():
            continue
        paths.extend(list_supported_images(folder))

    return sorted(paths, key=lambda path: (str(path.parent), path.name.lower()))


def build_output_paths(archive_ingest_dir: Path, source_path: Path) -> dict[str, Path]:
    stem = source_path.stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"{stem}-{timestamp}"

    archive_candidate = reserve_output_path(archive_ingest_dir / "working" / "cropped" / f"{base_name}-archive.jpg")
    enhanced_candidate = reserve_output_path(archive_ingest_dir / "working" / "enhanced" / f"{base_name}-share.jpg")
    preview_candidate = reserve_output_path(archive_ingest_dir / "working" / "enhanced" / f"{base_name}-preview.jpg")
    contact_sheet = reserve_output_path(archive_ingest_dir / "review" / "contact-sheets" / f"{base_name}-contact-sheet.jpg")
    notes_file = reserve_output_path(archive_ingest_dir / "review" / "notes" / f"{base_name}-image-review.txt")

    return {
        "archive": archive_candidate,
        "share": enhanced_candidate,
        "preview": preview_candidate,
        "contact_sheet": contact_sheet,
        "notes": notes_file,
    }


def main() -> None:
    script_path = Path(__file__).resolve()
    archive_ingest_dir = script_path.parent

    for folder in [
        archive_ingest_dir / "working" / "cropped",
        archive_ingest_dir / "working" / "enhanced",
        archive_ingest_dir / "review" / "contact-sheets",
        archive_ingest_dir / "review" / "notes",
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    print("\nShows I Saw - Ticket Image Processor v1.0\n")
    print("This tool creates review candidates only. It does not publish, overwrite originals, or update the live site.\n")

    source_path = choose_source_file(gather_source_images(archive_ingest_dir))
    print(f"\nSelected image: {source_path}\n")
    print("Processing image... please wait.\n")

    source_image = load_image(source_path)
    result = process_ticket_image(source_image)
    output_paths = build_output_paths(archive_ingest_dir, source_path)

    save_image(result.archive_image, output_paths["archive"])
    save_image(result.share_image, output_paths["share"])
    save_image(result.preview_image, output_paths["preview"])

    contact_sheet = build_contact_sheet(
        ContactSheetInputs(
            original=source_image,
            archive_candidate=result.archive_image,
            share_candidate=result.share_image,
            preview_candidate=result.preview_image,
            title="Shows I Saw - Image Processing Review",
            subtitle=f"Source: {source_path.name}",
        )
    )
    save_image(contact_sheet, output_paths["contact_sheet"])

    notes_text = build_processing_notes(
        source_path=source_path,
        archive_output=output_paths["archive"],
        share_output=output_paths["share"],
        preview_output=output_paths["preview"],
        contact_sheet_output=output_paths["contact_sheet"],
        result=result,
    )
    output_paths["notes"].write_text(notes_text, encoding="utf-8")

    print("Done.\n")
    print(f"Archive candidate:  {output_paths['archive']}")
    print(f"Share candidate:    {output_paths['share']}")
    print(f"Preview overlay:    {output_paths['preview']}")
    print(f"Contact sheet:      {output_paths['contact_sheet']}")
    print(f"Review notes:       {output_paths['notes']}")
    print("\nManual review is required before anything is copied into the live site.\n")


if __name__ == "__main__":
    main()
