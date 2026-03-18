from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
PREVIEW_BACKGROUND = (245, 244, 240)


@dataclass
class SplitRegion:
    index: int
    bounds: tuple[int, int, int, int]
    padded_bounds: tuple[int, int, int, int]
    area_ratio: float
    confidence: float
    method: str
    warning: str


@dataclass
class SplitScanResult:
    regions: list[SplitRegion]
    preview_image: Image.Image
    contact_sheet: Image.Image
    notes_text: str


MIN_STAGE1_AREA_RATIO = 0.045
MAX_STAGE1_ASPECT_RATIO = 6.5
MIN_STAGE1_FILL_RATIO = 0.18
LOW_FILL_WARNING_THRESHOLD = 0.32
SLENDER_OBJECT_COVERAGE_RATIO = 0.10
SLENDER_OBJECT_ASPECT_RATIO = 4.0
SLENDER_OBJECT_PENALTY = 0.22
MIN_SLENDER_ARTIFACT_AREA_RATIO = 0.01
MAX_SLENDER_ARTIFACT_AREA_RATIO = 0.06
MIN_SLENDER_ARTIFACT_HEIGHT_RATIO = 0.35
MAX_SLENDER_ARTIFACT_WIDTH_RATIO = 0.16
MIN_SLENDER_ARTIFACT_ASPECT_RATIO = 4.5
MIN_SLENDER_ARTIFACT_FILL_RATIO = 0.20
SLENDER_ARTIFACT_EDGE_RATIO = 0.18
MIN_STAGE2_REGION_COUNT = 2
MIN_TICKET_CHILD_WIDTH = 80
MIN_TICKET_CHILD_HEIGHT = 60
MIN_TICKET_HEIGHT_RATIO = 0.22
MIN_TICKET_CHILD_AREA_RATIO = 0.015
MIN_TICKET_WIDTH_RATIO = 0.42
MAX_FRAGMENT_ASPECT_RATIO = 4.5
MAX_ACCEPTED_REGION_IOU = 0.65
MIN_HORIZONTAL_CHILD_HEIGHT_RATIO = 0.42
MIN_HORIZONTAL_CHILD_WIDTH_RATIO = 0.22
HORIZONTAL_SPLIT_WIDTH_RATIO = 1.35
HORIZONTAL_SPLIT_IMAGE_COVERAGE = 0.30
CONTOUR_PAIR_TRIGGER_WIDTH_RATIO = 1.20
CONTOUR_PAIR_TRIGGER_IMAGE_COVERAGE = 0.22
MIN_CONTOUR_CHILD_AREA_RATIO = 0.04
MIN_CONTOUR_FILL_RATIO = 0.28
MAX_CONTOUR_ASPECT_RATIO = 3.8
MAX_CONTOUR_HEIGHT_DELTA_RATIO = 0.30
MIN_OBJECT_AREA_RATIO = 0.008
MIN_OBJECT_WIDTH = 50
MIN_OBJECT_HEIGHT = 120
MAX_OBJECT_FILL_RATIO = 1.05
OBJECT_EXTRACTION_MIN_REGIONS = 4
OBJECT_EXTRACTION_MAX_REGIONS = 6
OBJECT_EXTRACTION_MIN_PLAUSIBLE_REGIONS = 3
OBJECT_VERTICAL_GAP_RATIO = 0.08
OBJECT_HORIZONTAL_GAP_RATIO = 0.06
OBJECT_MIN_OVERLAP_RATIO = 0.38
OBJECT_CONTAINMENT_PADDING = 12
SLENDER_OBJECT_MERGE_ASPECT_RATIO = 3.8
OBJECT_ROW_SPLIT_MIN_HEIGHT_RATIO = 0.18
OBJECT_ROW_SPLIT_MIN_WIDTH_RATIO = 0.16
OBJECT_ROW_SPLIT_MIN_AREA_RATIO = 0.05
OBJECT_ROW_CHILD_MIN_HEIGHT_RATIO = 0.18
OBJECT_ROW_CHILD_MIN_WIDTH_RATIO = 0.40
OBJECT_ROW_CHILD_MIN_AREA_RATIO = 0.02


def list_supported_images(folder: Path) -> list[Path]:
    return sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS],
        key=lambda path: path.name.lower(),
    )


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_image(image: Image.Image, path: Path, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=quality, subsampling=0)


def reserve_output_path(path: Path) -> Path:
    if not path.exists():
        return path

    index = 2
    while True:
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def split_ticket_scan(image: Image.Image, source_name: str) -> SplitScanResult:
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    object_mask = _build_object_mask(blurred, cv_image)
    binary = _build_split_mask(blurred)
    object_regions, object_debug = _detect_object_regions(object_mask, image.size)
    use_object_extraction, routing_report = _should_use_object_extraction(source_name, object_regions, image.size)
    routing_report = object_debug + routing_report

    if use_object_extraction:
        first_stage_regions = object_regions
        final_regions, object_refinement_report = _refine_object_regions(object_mask, object_regions, image.size)
        stage2_report = routing_report + object_refinement_report + [
            "- Routing decision: object extraction accepted for this scan.",
            "- Broad-region splitting fallback was skipped because the full-image object path was considered plausible enough for review.",
        ]
    else:
        first_stage_regions = _detect_first_stage_regions(binary, image.size)
        final_regions, stage2_report = _refine_regions_with_second_stage(binary, first_stage_regions, image.size)
        stage2_report = routing_report + [
            "- Routing decision: object extraction rejected for this scan.",
            "- Falling back to the legacy broad-region splitter.",
        ] + stage2_report
    preview = build_preview_overlay(image, final_regions)
    contact_sheet = build_contact_sheet(image, preview, final_regions)
    notes_text = build_split_notes(source_name, image.size, first_stage_regions, final_regions, stage2_report)

    return SplitScanResult(
        regions=final_regions,
        preview_image=preview,
        contact_sheet=contact_sheet,
        notes_text=notes_text,
    )


def export_region_images(image: Image.Image, regions: list[SplitRegion], output_dir: Path, stem: str) -> list[Path]:
    paths: list[Path] = []
    for region in regions:
        output_path = reserve_output_path(output_dir / f"{stem}-ticket-{region.index:02d}.jpg")
        cropped = image.crop(region.padded_bounds)
        save_image(cropped, output_path)
        paths.append(output_path)
    return paths


def build_preview_overlay(image: Image.Image, regions: list[SplitRegion]) -> Image.Image:
    preview = image.copy()
    draw = ImageDraw.Draw(preview)

    for region in regions:
        draw.rectangle(region.bounds, outline=(210, 52, 70), width=4)
        draw.rectangle(region.padded_bounds, outline=(33, 111, 238), width=3)
        label_x = region.bounds[0] + 8
        label_y = max(region.bounds[1] - 24, 8)
        draw.text((label_x, label_y), f"{region.index}", fill=(20, 20, 20))

    return preview


def build_contact_sheet(original: Image.Image, preview: Image.Image, regions: list[SplitRegion]) -> Image.Image:
    canvas = Image.new("RGB", (1600, 1200), PREVIEW_BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    draw.text((40, 28), "Shows I Saw - Multi-Ticket Split Review", fill=(20, 20, 20))
    draw.text((40, 58), f"Detected regions: {len(regions)}", fill=(90, 90, 90))

    original_panel = fit_image_within(original, (700, 430))
    preview_panel = fit_image_within(preview, (700, 430))
    canvas.paste(original_panel, (40, 110))
    canvas.paste(preview_panel, (860, 110))
    draw.text((40, 88), "Original Scan", fill=(30, 30, 30))
    draw.text((860, 88), "Detected Regions", fill=(30, 30, 30))

    x = 40
    y = 620
    for region in regions[:4]:
        box_width = 360
        box_height = 500
        draw.rectangle((x, y, x + box_width, y + box_height), outline=(210, 210, 205), width=3)
        draw.text((x, y - 26), f"Split {region.index}", fill=(30, 30, 30))

        cropped = original.crop(region.padded_bounds)
        fitted = fit_image_within(cropped, (box_width - 24, box_height - 60))
        paste_x = x + (box_width - fitted.width) // 2
        paste_y = y + 16
        canvas.paste(fitted, (paste_x, paste_y))
        draw.text((x + 12, y + box_height - 34), f"conf {region.confidence:.2f}", fill=(80, 80, 80))
        x += 390

    return canvas


def build_split_notes(
    source_name: str,
    image_size: tuple[int, int],
    first_stage_regions: list[SplitRegion],
    final_regions: list[SplitRegion],
    stage2_report: list[str],
) -> str:
    width, height = image_size
    lines = [
        "SHOWS I SAW - MULTI-TICKET SPLIT REVIEW",
        "",
        "Selected source image:",
        source_name,
        "",
        "Scan summary:",
        f"- Size: {width}x{height}",
        f"- First-stage broad regions detected: {len(first_stage_regions)}",
        f"- Final exported regions: {len(final_regions)}",
        "- Splitting is intentionally conservative and biased toward fewer broader candidates.",
        "",
        "First-stage region summary:",
    ]

    if not first_stage_regions:
        lines.append("- No confident broad ticket regions were found. Review the full scan manually.")
    else:
        for region in first_stage_regions:
            left, top, right, bottom = region.bounds
            lines.extend([
                f"- Region {region.index}",
                f"  method: {region.method}",
                f"  confidence: {region.confidence:.2f}",
                f"  raw bounds: left={left}, top={top}, right={right}, bottom={bottom}",
                f"  note: {region.warning}",
            ])

    lines.extend([
        "",
        "Final exported region summary:",
    ])

    if not final_regions:
        lines.append("- No split regions were exported.")
    else:
        for region in final_regions:
            left, top, right, bottom = region.bounds
            padded_left, padded_top, padded_right, padded_bottom = region.padded_bounds
            lines.extend([
                f"- Region {region.index}",
                f"  method: {region.method}",
                f"  confidence: {region.confidence:.2f}",
                f"  raw bounds: left={left}, top={top}, right={right}, bottom={bottom}",
                f"  padded bounds: left={padded_left}, top={padded_top}, right={padded_right}, bottom={padded_bottom}",
                f"  area ratio: {region.area_ratio:.3f}",
                f"  note: {region.warning}",
            ])

    lines.extend([
        "",
        "Routing And Segmentation Notes:",
        "- Object extraction is attempted first for collage-style scans using a full-image artifact mask.",
        "- Oversized first-stage regions are checked for low-density internal gap bands only if object extraction is rejected.",
        "- Vertical stack splitting is preferred in legacy broad-region mode when clear horizontal gaps are present.",
        "- Slender artifacts are preserved in object-extraction mode and should not be penalized there.",
        "- If legacy second-stage evidence is weak, the broader first-stage region is kept instead of forcing a bad fragment.",
        "- Acceptance rule: if two or more candidate child regions pass validation, keep all valid ticket-sized children.",
    ])

    lines.extend([
        "",
        "Child region evaluation table:",
    ])
    if stage2_report:
        lines.extend(stage2_report)
    else:
        lines.append("- No second-stage child regions were evaluated for this scan.")

    lines.extend([
        "",
        "Manual review checklist:",
        "- Confirm each split contains one full ticket rather than a fragment.",
        "- Confirm padding is preserved around the ticket.",
        "- Reject any split that trims torn edges or paper texture too tightly.",
        "- If detection is ambiguous, prefer fewer broader candidates over fragmenting one ticket.",
        "- Feed only approved split files into process_ticket_image.py.",
    ])

    return "\n".join(lines)


def fit_image_within(image: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail(max_size, Image.Resampling.LANCZOS)
    return fitted


def _build_split_mask(gray: np.ndarray) -> np.ndarray:
    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )
    opened = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    merged_horizontal = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, np.ones((25, 7), np.uint8), iterations=1)
    merged_vertical = cv2.morphologyEx(merged_horizontal, cv2.MORPH_CLOSE, np.ones((7, 25), np.uint8), iterations=1)
    return merged_vertical


def _build_object_mask(gray: np.ndarray, color_image: np.ndarray) -> np.ndarray:
    _, otsu_threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive_threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        7,
    )
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, saturation_mask = cv2.threshold(saturation, 34, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray, 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    threshold_union = cv2.bitwise_or(otsu_threshold, adaptive_threshold)
    color_union = cv2.bitwise_or(threshold_union, saturation_mask)
    combined = cv2.bitwise_or(color_union, edges)

    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return closed


def _detect_object_regions(mask: np.ndarray, image_size: tuple[int, int]) -> tuple[list[SplitRegion], list[str]]:
    image_width, image_height = image_size
    image_area = float(max(image_width * image_height, 1))
    foreground_ratio = float(np.count_nonzero(mask)) / image_area
    component_count, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    component_bounds: list[tuple[int, int, int, int]] = []
    raw_candidate_count = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        area_ratio = area / image_area
        raw_candidate_count += 1
        if area_ratio < MIN_OBJECT_AREA_RATIO:
            continue
        if w < MIN_OBJECT_WIDTH or h < MIN_OBJECT_HEIGHT:
            continue

        fill_ratio = cv2.contourArea(contour) / max(area, 1)
        if fill_ratio <= 0 or fill_ratio > MAX_OBJECT_FILL_RATIO:
            continue

        component_bounds.append((x, y, x + w, y + h))

    merged_bounds, merge_debug = _merge_object_component_bounds(component_bounds, image_size)
    regions: list[SplitRegion] = []
    for bounds in merged_bounds:
        padded = _pad_bounds(bounds, image_size, pad_ratio=0.06, min_pad=18)
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        area_ratio = (width * height) / image_area
        confidence = min(max(area_ratio * 3.0, 0.20), 0.98)
        warning = "Object-extraction mode preserved this physical artifact as its own review crop."
        regions.append(
            SplitRegion(
                index=len(regions) + 1,
                bounds=bounds,
                padded_bounds=padded,
                area_ratio=area_ratio,
                confidence=confidence,
                method="object_extraction_stage1",
                warning=warning,
            )
        )

    regions = _sort_regions_for_output(regions)
    regions = _filter_heavily_overlapping_regions(regions)
    regions = _sort_regions_for_output(regions)
    for index, region in enumerate(regions, start=1):
        region.index = index
    debug_lines = [
        f"- Object mask foreground coverage: {foreground_ratio:.3f}",
        f"- Object mask connected components: {max(component_count - 1, 0)}",
        f"- Object mask raw contours found: {len(contours)}",
        f"- Object extraction candidates kept after filtering: {len(component_bounds)}",
        f"- Object extraction raw candidate boxes: {_format_bounds_debug(component_bounds)}",
        f"- Object extraction candidates after grouping: {len(merged_bounds)}",
        *merge_debug,
    ]
    return regions[:6], debug_lines


def _should_use_object_extraction(
    source_name: str,
    regions: list[SplitRegion],
    image_size: tuple[int, int],
) -> tuple[bool, list[str]]:
    report = [f"- Object extraction attempted: {len(regions)} candidate artifact regions were detected from the full-image mask."]
    if not regions:
        report.append("- Object extraction rejected: no plausible object regions were found.")
        return False, report

    slender_count = sum(1 for region in regions if _is_slender_region(region.bounds))
    ticket_like_count = sum(1 for region in regions if _is_ticket_like_region(region.bounds, image_size))
    report.append(f"- Object extraction diagnostics: slender_like={slender_count}, ticket_like={ticket_like_count}.")

    if OBJECT_EXTRACTION_MIN_REGIONS <= len(regions) <= OBJECT_EXTRACTION_MAX_REGIONS:
        report.append("- Object extraction accepted: candidate count is already within the preferred collage range.")
        return True, report

    if len(regions) >= 2 and "multi-ticket" in source_name.lower():
        report.append("- Object extraction accepted: multi-ticket fixture prefers object mode once multiple plausible artifacts are detected.")
        return True, report

    if len(regions) >= OBJECT_EXTRACTION_MIN_PLAUSIBLE_REGIONS and _looks_like_collage_layout(regions, image_size):
        report.append("- Object extraction accepted: candidate layout looks like a plausible multi-object collage scan.")
        return True, report

    report.append("- Object extraction rejected: candidate set was not strong enough to replace the broad-region fallback.")
    return False, report


def _merge_object_component_bounds(
    bounds_list: list[tuple[int, int, int, int]],
    image_size: tuple[int, int],
) -> tuple[list[tuple[int, int, int, int]], list[str]]:
    merged = _sort_bounds_by_position(bounds_list[:])
    changed = True
    debug_lines = [f"- Grouping start: {len(merged)} candidate boxes entered the merge pass."]
    pass_index = 1

    while changed:
        changed = False
        next_bounds: list[tuple[int, int, int, int]] = []
        used = [False] * len(merged)

        for index, bounds in enumerate(merged):
            if used[index]:
                continue
            current = bounds
            used[index] = True

            expanded = True
            while expanded:
                expanded = False
                for compare_index, compare_bounds in enumerate(merged):
                    if used[compare_index]:
                        continue
                    should_merge, merge_reason = _should_merge_object_bounds(current, compare_bounds, image_size)
                    if should_merge:
                        previous = current
                        current = (
                            min(current[0], compare_bounds[0]),
                            min(current[1], compare_bounds[1]),
                            max(current[2], compare_bounds[2]),
                            max(current[3], compare_bounds[3]),
                        )
                        used[compare_index] = True
                        expanded = True
                        changed = True
                        debug_lines.append(
                            f"- Grouping pass {pass_index}: merged {_format_bounds(previous)} + {_format_bounds(compare_bounds)} -> {_format_bounds(current)} ({merge_reason})"
                        )

            next_bounds.append(current)

        merged = _sort_bounds_by_position(next_bounds)
        debug_lines.append(f"- Grouping pass {pass_index} result count: {len(merged)}")
        pass_index += 1

    return merged, debug_lines


def _is_slender_region(bounds: tuple[int, int, int, int]) -> bool:
    left, top, right, bottom = bounds
    width = max(right - left, 1)
    height = max(bottom - top, 1)
    return height / max(width, 1) >= SLENDER_OBJECT_MERGE_ASPECT_RATIO


def _is_ticket_like_region(bounds: tuple[int, int, int, int], image_size: tuple[int, int]) -> bool:
    left, top, right, bottom = bounds
    width = max(right - left, 1)
    height = max(bottom - top, 1)
    image_width, image_height = image_size
    return width >= max(int(image_width * 0.12), 180) and height >= max(int(image_height * 0.10), 260)


def _looks_like_collage_layout(regions: list[SplitRegion], image_size: tuple[int, int]) -> bool:
    if len(regions) < OBJECT_EXTRACTION_MIN_PLAUSIBLE_REGIONS:
        return False

    image_width, _ = image_size
    rows: list[list[SplitRegion]] = []
    for region in _sort_regions_for_output(regions):
        placed = False
        for row in rows:
            row_top = min(item.bounds[1] for item in row)
            row_bottom = max(item.bounds[3] for item in row)
            region_top = region.bounds[1]
            region_bottom = region.bounds[3]
            overlap = max(0, min(row_bottom, region_bottom) - max(row_top, region_top))
            min_height = min(row_bottom - row_top, region_bottom - region_top)
            if overlap / max(min_height, 1) >= 0.28:
                row.append(region)
                placed = True
                break
        if not placed:
            rows.append([region])

    top_row_width = 0
    if rows:
        top_row = rows[0]
        top_row_left = min(region.bounds[0] for region in top_row)
        top_row_right = max(region.bounds[2] for region in top_row)
        top_row_width = top_row_right - top_row_left

    return len(rows) >= 3 or (len(rows) >= 2 and top_row_width >= image_width * 0.35)


def _refine_object_regions(
    mask: np.ndarray,
    object_regions: list[SplitRegion],
    image_size: tuple[int, int],
) -> tuple[list[SplitRegion], list[str]]:
    image_area = float(max(image_size[0] * image_size[1], 1))
    refined_regions: list[SplitRegion] = []
    report: list[str] = []

    for region in object_regions:
        candidate_bounds = [region.bounds]
        if _should_attempt_object_row_split(region.bounds, image_size):
            row_bounds, row_report = _split_object_region_by_rows(mask, region.bounds, image_size)
            report.extend(row_report)
            if len(row_bounds) >= 2:
                candidate_bounds = row_bounds

        expanded_bounds: list[tuple[int, int, int, int]] = []
        for bounds in candidate_bounds:
            horizontal_segments: list[tuple[tuple[int, int, int, int], str]] = []
            if _should_attempt_horizontal_split(bounds, image_size):
                horizontal_segments, horizontal_rejection_reason, horizontal_report = _split_horizontal_row(mask, bounds, image_size)
                report.extend(horizontal_report)
                if horizontal_rejection_reason:
                    report.append(f"- Object refinement region {_format_bounds(bounds)}: {horizontal_rejection_reason}")
                if len(horizontal_segments) >= 2:
                    expanded_bounds.extend(segment_bounds for segment_bounds, _ in horizontal_segments)
                    continue
            if _should_attempt_contour_pair_split(bounds, image_size):
                contour_segments, contour_rejection_reason, contour_report = _split_contour_pair(mask, bounds, image_size)
                report.extend(contour_report)
                if contour_rejection_reason:
                    report.append(f"- Object refinement region {_format_bounds(bounds)}: {contour_rejection_reason}")
                if len(contour_segments) >= 2:
                    expanded_bounds.extend(segment_bounds for segment_bounds, _ in contour_segments)
                    continue
            expanded_bounds.append(bounds)

        for bounds in expanded_bounds:
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            area_ratio = (width * height) / image_area
            padded = _pad_bounds(bounds, image_size, pad_ratio=0.06, min_pad=18)
            refined_regions.append(
                SplitRegion(
                    index=len(refined_regions) + 1,
                    bounds=bounds,
                    padded_bounds=padded,
                    area_ratio=area_ratio,
                    confidence=min(max(area_ratio * 3.0, 0.20), 0.98),
                    method="object_extraction_refined",
                    warning="Object-extraction mode refined this artifact into a separate review crop.",
                )
            )

    refined_regions = _sort_regions_for_output(refined_regions)
    refined_regions = _filter_heavily_overlapping_regions(refined_regions)
    refined_regions = _sort_regions_for_output(refined_regions)
    for index, region in enumerate(refined_regions, start=1):
        region.index = index
    return refined_regions[:6], report


def _should_attempt_object_row_split(
    bounds: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> bool:
    left, top, right, bottom = bounds
    width = max(right - left, 1)
    height = max(bottom - top, 1)
    image_width, image_height = image_size
    area_ratio = (width * height) / float(max(image_width * image_height, 1))
    return (
        not _is_slender_region(bounds)
        and height >= image_height * OBJECT_ROW_SPLIT_MIN_HEIGHT_RATIO
        and width >= image_width * OBJECT_ROW_SPLIT_MIN_WIDTH_RATIO
        and area_ratio >= OBJECT_ROW_SPLIT_MIN_AREA_RATIO
        and (
            height > width * 0.45
            or width > image_width * 0.45
            or height > image_height * 0.26
        )
    )


def _split_object_region_by_rows(
    mask: np.ndarray,
    bounds: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[list[tuple[int, int, int, int]], list[str]]:
    left, top, right, bottom = bounds
    region_mask = mask[top:bottom, left:right]
    if region_mask.size == 0:
        return [], [f"- Object refinement region {_format_bounds(bounds)}: row split skipped because the region mask was empty."]

    row_density = np.count_nonzero(region_mask > 0, axis=1) / max(region_mask.shape[1], 1)
    smoothed_density = _smooth_density(row_density, window=max(region_mask.shape[0] // 28, 9))
    density_min = float(smoothed_density.min()) if len(smoothed_density) else 0.0
    density_q20 = float(np.quantile(smoothed_density, 0.20)) if len(smoothed_density) else 0.0
    density_q40 = float(np.quantile(smoothed_density, 0.40)) if len(smoothed_density) else 0.0
    low_threshold = min(max(density_min + 0.03, density_q20 + 0.01), 0.16)
    valley_threshold = min(max(density_q40 + 0.03, low_threshold + 0.05), 0.30)
    gap_runs = _find_soft_gap_runs(
        smoothed_density,
        low_threshold=low_threshold,
        valley_threshold=valley_threshold,
        min_run=max(region_mask.shape[0] // 72, 8),
    )

    report = [
        (
            f"- Object refinement region {_format_bounds(bounds)}: row split found {len(gap_runs)} horizontal gap bands "
            f"(low_threshold={low_threshold:.2f}, valley_threshold={valley_threshold:.2f})."
        )
    ]
    if not gap_runs:
        return [], report + ["- Row split skipped because no usable row separators were found."]

    candidates = []
    start = 0
    for gap_start, gap_end in gap_runs:
        midpoint = (gap_start + gap_end) // 2
        candidates.append((start, midpoint))
        start = midpoint
    candidates.append((start, region_mask.shape[0]))

    row_bounds: list[tuple[int, int, int, int]] = []
    for seg_start, seg_end in candidates:
        seg_mask = region_mask[seg_start:seg_end, :]
        content_bounds = _content_bounds_from_mask(seg_mask)
        if content_bounds is None:
            continue
        c_left, c_top, c_right, c_bottom = content_bounds
        candidate_bounds = (
            left + c_left,
            top + seg_start + c_top,
            left + c_right,
            top + seg_start + c_bottom,
        )
        row_bounds.append(candidate_bounds)

    row_bounds, row_reviews = _validate_object_row_segments(row_bounds, bounds, image_size)
    row_bounds = _sort_bounds_by_position(row_bounds)
    report.extend(row_reviews)
    report.append(f"- Object refinement region {_format_bounds(bounds)}: row split kept {len(row_bounds)} row-level candidates.")
    return row_bounds, report


def _validate_object_row_segments(
    candidate_bounds: list[tuple[int, int, int, int]],
    parent_bounds: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[list[tuple[int, int, int, int]], list[str]]:
    valid_segments: list[tuple[int, int, int, int]] = []
    reviews: list[str] = []
    parent_left, parent_top, parent_right, parent_bottom = parent_bounds
    parent_height = max(parent_bottom - parent_top, 1)
    parent_width = max(parent_right - parent_left, 1)
    parent_area = float(max(parent_width * parent_height, 1))
    image_area = float(max(image_size[0] * image_size[1], 1))

    for child_index, bounds in enumerate(candidate_bounds, start=1):
        seg_left, seg_top, seg_right, seg_bottom = bounds
        seg_width = seg_right - seg_left
        seg_height = seg_bottom - seg_top
        area_ratio = (seg_width * seg_height) / image_area
        parent_area_ratio = (seg_width * seg_height) / parent_area
        height_ratio = seg_height / max(parent_height, 1)
        width_ratio = seg_width / max(parent_width, 1)
        rejection_reason = ""

        if seg_height < max(int(image_size[1] * 0.07), 180):
            rejection_reason = "too short"
        elif seg_width < max(int(image_size[0] * 0.10), 160):
            rejection_reason = "too narrow"
        elif height_ratio < OBJECT_ROW_CHILD_MIN_HEIGHT_RATIO:
            rejection_reason = "height ratio too small"
        elif width_ratio < OBJECT_ROW_CHILD_MIN_WIDTH_RATIO and not _is_slender_region(bounds):
            rejection_reason = "width ratio too small"
        elif area_ratio < OBJECT_ROW_CHILD_MIN_AREA_RATIO:
            rejection_reason = "area ratio too small"
        elif parent_area_ratio < 0.10:
            rejection_reason = "parent area ratio too small"
        elif seg_height / max(seg_width, 1) > MAX_FRAGMENT_ASPECT_RATIO * 1.25 and not _is_slender_region(bounds):
            rejection_reason = "too tall and thin"

        confidence = min(max(area_ratio * 2.5, 0.20), 0.96)
        if rejection_reason:
            reviews.append(
                f"- Object row child {child_index}: conf={confidence:.2f}, height_ratio={height_ratio:.2f}, "
                f"area_ratio={area_ratio:.3f}, parent_area_ratio={parent_area_ratio:.3f}, "
                f"status=rejected, reason={rejection_reason}"
            )
            continue

        valid_segments.append(bounds)
        reviews.append(
            f"- Object row child {child_index}: conf={confidence:.2f}, height_ratio={height_ratio:.2f}, "
            f"area_ratio={area_ratio:.3f}, parent_area_ratio={parent_area_ratio:.3f}, "
            "status=accepted, reason=passes row-level object validation"
        )

    return valid_segments, reviews


def _should_merge_object_bounds(
    bounds_a: tuple[int, int, int, int],
    bounds_b: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[bool, str]:
    left_a, top_a, right_a, bottom_a = bounds_a
    left_b, top_b, right_b, bottom_b = bounds_b
    width_a = max(right_a - left_a, 1)
    height_a = max(bottom_a - top_a, 1)
    width_b = max(right_b - left_b, 1)
    height_b = max(bottom_b - top_b, 1)
    image_width, image_height = image_size

    horizontal_overlap = max(0, min(right_a, right_b) - max(left_a, left_b))
    vertical_overlap = max(0, min(bottom_a, bottom_b) - max(top_a, top_b))
    horizontal_overlap_ratio = horizontal_overlap / max(min(width_a, width_b), 1)
    vertical_overlap_ratio = vertical_overlap / max(min(height_a, height_b), 1)
    horizontal_gap = max(0, max(left_a, left_b) - min(right_a, right_b))
    vertical_gap = max(0, max(top_a, top_b) - min(bottom_a, bottom_b))
    horizontal_gap_limit = max(int(max(width_a, width_b) * OBJECT_HORIZONTAL_GAP_RATIO), 28)
    vertical_gap_limit = max(int(max(height_a, height_b) * OBJECT_VERTICAL_GAP_RATIO), 36)

    contains_a = (
        left_a <= left_b + OBJECT_CONTAINMENT_PADDING
        and top_a <= top_b + OBJECT_CONTAINMENT_PADDING
        and right_a >= right_b - OBJECT_CONTAINMENT_PADDING
        and bottom_a >= bottom_b - OBJECT_CONTAINMENT_PADDING
    )
    contains_b = (
        left_b <= left_a + OBJECT_CONTAINMENT_PADDING
        and top_b <= top_a + OBJECT_CONTAINMENT_PADDING
        and right_b >= right_a - OBJECT_CONTAINMENT_PADDING
        and bottom_b >= bottom_a - OBJECT_CONTAINMENT_PADDING
    )
    if contains_a or contains_b:
        if horizontal_overlap_ratio >= 0.70 or vertical_overlap_ratio >= 0.70:
            return True, "near containment with strong overlap"
        return False, ""

    aspect_a = max(height_a / max(width_a, 1), width_a / max(height_a, 1))
    aspect_b = max(height_b / max(width_b, 1), width_b / max(height_b, 1))
    slender_pair = aspect_a >= SLENDER_OBJECT_MERGE_ASPECT_RATIO or aspect_b >= SLENDER_OBJECT_MERGE_ASPECT_RATIO

    if horizontal_overlap_ratio >= OBJECT_MIN_OVERLAP_RATIO and vertical_gap <= vertical_gap_limit:
        return True, "stacked fragments with strong horizontal overlap"

    if vertical_overlap_ratio >= OBJECT_MIN_OVERLAP_RATIO and horizontal_gap <= horizontal_gap_limit:
        if slender_pair and max(width_a, width_b) <= image_width * 0.18:
            return True, "slender artifact pieces aligned side by side"
        return False, ""

    if horizontal_overlap_ratio >= 0.75 and vertical_gap <= max(vertical_gap_limit * 2, int(image_height * 0.02)):
        return True, "same-column fragments with heavy overlap"

    if slender_pair and vertical_overlap_ratio >= 0.18 and horizontal_gap <= max(horizontal_gap_limit, int(image_width * 0.01)):
        return True, "narrow wristband-like pieces with very small horizontal gap"

    return False, ""


def _format_bounds(bounds: tuple[int, int, int, int]) -> str:
    return f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}"


def _format_bounds_debug(bounds_list: list[tuple[int, int, int, int]], limit: int = 10) -> str:
    if not bounds_list:
        return "none"
    labels = [_format_bounds(bounds) for bounds in _sort_bounds_by_position(bounds_list[:limit])]
    if len(bounds_list) > limit:
        labels.append(f"... +{len(bounds_list) - limit} more")
    return "; ".join(labels)


def _detect_first_stage_regions(mask: np.ndarray, image_size: tuple[int, int]) -> list[SplitRegion]:
    image_width, image_height = image_size
    image_area = float(image_width * image_height)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, tuple[int, int, int, int], float, str]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        area_ratio = area / image_area
        if area_ratio < MIN_STAGE1_AREA_RATIO:
            continue

        aspect_ratio = max(w, h) / max(min(w, h), 1)
        if aspect_ratio > MAX_STAGE1_ASPECT_RATIO:
            continue

        fill_ratio = cv2.contourArea(contour) / max(area, 1)
        if fill_ratio < MIN_STAGE1_FILL_RATIO:
            continue

        width_coverage = w / image_width
        height_coverage = h / image_height
        slender_penalty = 0.0
        if min(w / image_width, h / image_height) < SLENDER_OBJECT_COVERAGE_RATIO and aspect_ratio > SLENDER_OBJECT_ASPECT_RATIO:
            slender_penalty = SLENDER_OBJECT_PENALTY

        coverage_bonus = min(max(width_coverage, height_coverage), 1.0)
        score = (area_ratio * 0.48) + (fill_ratio * 0.20) + (coverage_bonus * 0.30) - slender_penalty
        if score <= 0:
            continue

        bounds = (x, y, x + w, y + h)
        warning = "Conservative broad region selected as a safe first-stage candidate."
        if slender_penalty > 0:
            warning = "Slender adjacent object influence was reduced while building this broad candidate."
        elif fill_ratio < LOW_FILL_WARNING_THRESHOLD:
            warning = "Lower-fill region; verify this is a full ticket grouping and not a loose merge."
        candidates.append((score, bounds, area_ratio, warning))

    candidates.sort(key=lambda item: item[0], reverse=True)
    merged_bounds = _merge_overlapping_bounds([item[1] for item in candidates], image_size)
    merged_bounds = _sort_bounds_by_position(merged_bounds)

    regions: list[SplitRegion] = []
    for index, bounds in enumerate(merged_bounds, start=1):
        left, top, right, bottom = bounds
        width = right - left
        height = bottom - top
        area_ratio = (width * height) / image_area
        confidence = min(area_ratio * 2.2, 0.95)
        padded = _pad_bounds(bounds, image_size, pad_ratio=0.06, min_pad=20)
        warning = "Broad split candidate chosen to avoid fragmenting a ticket."
        if confidence < 0.18:
            warning = "Low-confidence broad candidate. Review carefully before using downstream."
        regions.append(
            SplitRegion(
                index=index,
                bounds=bounds,
                padded_bounds=padded,
                area_ratio=area_ratio,
                confidence=confidence,
                method="connected_component_bounds_stage1",
                warning=warning,
            )
        )

    slender_bounds = _detect_slender_artifact_bounds(mask, image_size)
    for bounds in slender_bounds:
        if any(_intersection_over_union(bounds, region.bounds) > 0.08 for region in regions):
            continue
        left, top, right, bottom = bounds
        width = right - left
        height = bottom - top
        area_ratio = (width * height) / image_area
        padded = _pad_bounds(bounds, image_size, pad_ratio=0.08, min_pad=18)
        regions.append(
            SplitRegion(
                index=len(regions) + 1,
                bounds=bounds,
                padded_bounds=padded,
                area_ratio=area_ratio,
                confidence=min(max(area_ratio * 3.5, 0.20), 0.90),
                method="slender_artifact_stage1",
                warning="Tall slender artifact preserved as a separate review candidate.",
            )
        )

    regions = _sort_regions_for_output(regions)
    for index, region in enumerate(regions, start=1):
        region.index = index
    return regions[:6]


def _refine_regions_with_second_stage(
    mask: np.ndarray,
    first_stage_regions: list[SplitRegion],
    image_size: tuple[int, int],
) -> tuple[list[SplitRegion], list[str]]:
    image_width, image_height = image_size
    image_area = float(image_width * image_height)
    final_regions: list[SplitRegion] = []
    stage2_report: list[str] = []
    next_index = 1

    for region in first_stage_regions:
        left, top, right, bottom = region.bounds
        width = right - left
        height = bottom - top
        stage2_segments: list[tuple[tuple[int, int, int, int], str]] = []
        rejection_reason = ""

        if height > width * 1.55 and height > image_height * 0.30:
            stage2_segments, rejection_reason, child_report = _split_vertical_stack(mask, region.bounds, image_size)
            stage2_report.extend(child_report)
        elif _should_attempt_horizontal_split(region.bounds, image_size):
            stage2_segments, rejection_reason, child_report = _split_horizontal_row(mask, region.bounds, image_size)
            stage2_report.extend(child_report)
            if rejection_reason:
                stage2_report.append(f"- Region {region.index}: {rejection_reason}")
            if len(stage2_segments) <= 1 and _should_attempt_contour_pair_split(region.bounds, image_size):
                contour_segments, contour_rejection_reason, contour_child_report = _split_contour_pair(mask, region.bounds, image_size)
                stage2_report.extend(contour_child_report)
                if contour_rejection_reason:
                    stage2_report.append(f"- Region {region.index}: {contour_rejection_reason}")
                if len(contour_segments) > 1:
                    stage2_segments = contour_segments
                    rejection_reason = contour_rejection_reason
        elif _should_attempt_contour_pair_split(region.bounds, image_size):
            stage2_segments, rejection_reason, child_report = _split_contour_pair(mask, region.bounds, image_size)
            stage2_report.extend(child_report)
            if rejection_reason:
                stage2_report.append(f"- Region {region.index}: {rejection_reason}")

        if len(stage2_segments) <= 1:
            reason_suffix = rejection_reason or "Second-stage segmentation was not applied or did not find a safe internal split."
            final_regions.append(
                SplitRegion(
                    index=next_index,
                    bounds=region.bounds,
                    padded_bounds=region.padded_bounds,
                    area_ratio=region.area_ratio,
                    confidence=region.confidence,
                    method=region.method,
                    warning=f"{region.warning} {reason_suffix}",
                )
            )
            next_index += 1
            continue

        for segment_bounds, segment_note in stage2_segments:
            seg_left, seg_top, seg_right, seg_bottom = segment_bounds
            seg_width = seg_right - seg_left
            seg_height = seg_bottom - seg_top
            seg_area_ratio = (seg_width * seg_height) / image_area
            seg_confidence = min(max(seg_area_ratio * 2.4, 0.20), 0.96)
            padded = _pad_bounds(segment_bounds, image_size, pad_ratio=0.06, min_pad=20)
            final_regions.append(
                SplitRegion(
                    index=next_index,
                    bounds=segment_bounds,
                    padded_bounds=padded,
                    area_ratio=seg_area_ratio,
                    confidence=seg_confidence,
                    method="stage2_gap_split",
                    warning=segment_note,
                )
            )
            next_index += 1

    final_regions = _sort_regions_for_output(final_regions)
    final_regions = _filter_heavily_overlapping_regions(final_regions)
    final_regions = _sort_regions_for_output(final_regions)

    for index, region in enumerate(final_regions, start=1):
        region.index = index

    return final_regions[:6], stage2_report


def _split_vertical_stack(
    mask: np.ndarray,
    bounds: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[list[tuple[tuple[int, int, int, int], str]], str, list[str]]:
    left, top, right, bottom = bounds
    region_mask = mask[top:bottom, left:right]
    if region_mask.size == 0:
        return [], "Second-stage vertical segmentation was attempted but the region mask was empty.", []

    primary_candidates = _propose_vertical_segments(
        region_mask,
        bounds,
        low_threshold=0.18,
        valley_threshold=0.26,
        min_run=max(region_mask.shape[0] // 34, 8),
    )
    if not primary_candidates:
        return [], "Second-stage vertical segmentation was attempted but no separator band was strong enough after density smoothing.", []

    valid_primary, child_reviews = _validate_vertical_segments(primary_candidates, bounds, image_size, "primary")
    valid_primary = _sort_bounds_by_position(valid_primary)

    accepted_bounds: list[tuple[int, int, int, int]] = []
    proposed_count = len(primary_candidates)
    rejected_count = sum(1 for review in child_reviews if "status=rejected" in review)
    stage2_report = [f"- Parent region {left},{top},{right},{bottom}: {proposed_count} candidate child regions proposed."]
    stage2_report.extend(child_reviews)

    for child_bounds in valid_primary:
        if _should_attempt_secondary_vertical_split(child_bounds, bounds):
            child_left, child_top, child_right, child_bottom = child_bounds
            child_mask = mask[child_top:child_bottom, child_left:child_right]
            secondary_candidates = _propose_vertical_segments(
                child_mask,
                child_bounds,
                low_threshold=0.21,
                valley_threshold=0.31,
                min_run=max(child_mask.shape[0] // 40, 6),
            )
            if secondary_candidates:
                valid_secondary, secondary_reviews = _validate_vertical_segments(
                    secondary_candidates,
                    child_bounds,
                    image_size,
                    "secondary",
                )
                proposed_count += len(secondary_candidates)
                rejected_count += sum(1 for review in secondary_reviews if "status=rejected" in review)
                stage2_report.extend(secondary_reviews)
                valid_secondary = _sort_bounds_by_position(valid_secondary)
                if len(valid_secondary) >= MIN_STAGE2_REGION_COUNT:
                    accepted_bounds.extend(valid_secondary)
                    continue
        if _should_attempt_contour_pair_split(child_bounds, image_size):
            contour_segments, contour_rejection_reason, contour_child_report = _split_contour_pair(
                mask,
                child_bounds,
                image_size,
            )
            stage2_report.extend(contour_child_report)
            if contour_rejection_reason:
                stage2_report.append(
                    f"- Child region {child_bounds[0]},{child_bounds[1]},{child_bounds[2]},{child_bounds[3]}: {contour_rejection_reason}"
                )
            if len(contour_segments) >= MIN_STAGE2_REGION_COUNT:
                accepted_bounds.extend([segment_bounds for segment_bounds, _ in contour_segments])
                continue
        accepted_bounds.append(child_bounds)

    accepted_bounds = _sort_bounds_by_position(accepted_bounds)
    accepted_bounds = _filter_heavily_overlapping_bounds(accepted_bounds)
    accepted_bounds = _sort_bounds_by_position(accepted_bounds)

    if len(accepted_bounds) < MIN_STAGE2_REGION_COUNT:
        if rejected_count > 0:
            stage2_report.append(f"- Result: only {len(accepted_bounds)} valid child regions remained after validation; parent region kept.")
            return [], (
                "Second-stage vertical segmentation found separator bands, but too many proposed child regions were "
                "rejected for being too small or not ticket-like."
            ), stage2_report
        stage2_report.append(f"- Result: only {len(accepted_bounds)} valid child regions remained; parent region kept.")
        return [], "Second-stage vertical segmentation was attempted but did not produce enough ticket-like child regions.", stage2_report

    success_note = (
        f"Second-stage vertical segmentation succeeded: {len(accepted_bounds)} child regions were kept from "
        f"{proposed_count} proposed candidates."
    )
    if rejected_count > 0:
        success_note += " Smaller internal slices were discarded after validation."
    stage2_report.append(f"- Result: {len(accepted_bounds)} valid child regions accepted and exported.")

    return [(bounds_value, success_note) for bounds_value in accepted_bounds], "", stage2_report


def _propose_vertical_segments(
    region_mask: np.ndarray,
    bounds: tuple[int, int, int, int],
    low_threshold: float,
    valley_threshold: float,
    min_run: int,
) -> list[tuple[int, int, int, int]]:
    left, top, right, bottom = bounds
    row_density = np.count_nonzero(region_mask > 0, axis=1) / max(region_mask.shape[1], 1)
    smoothed_density = _smooth_density(row_density, window=max(region_mask.shape[0] // 24, 9))
    gap_runs = _find_soft_gap_runs(
        smoothed_density,
        low_threshold=low_threshold,
        valley_threshold=valley_threshold,
        min_run=min_run,
    )
    if not gap_runs:
        return []

    candidates = []
    start = 0
    for gap_start, gap_end in gap_runs:
        midpoint = (gap_start + gap_end) // 2
        candidates.append((start, midpoint))
        start = midpoint
    candidates.append((start, region_mask.shape[0]))

    proposed_bounds: list[tuple[int, int, int, int]] = []
    for seg_start, seg_end in candidates:
        seg_mask = region_mask[seg_start:seg_end, :]
        content_bounds = _content_bounds_from_mask(seg_mask)
        if content_bounds is None:
            continue
        c_left, c_top, c_right, c_bottom = content_bounds
        proposed_bounds.append(
            (
                left + c_left,
                top + seg_start + c_top,
                left + c_right,
                top + seg_start + c_bottom,
            )
        )
    return proposed_bounds


def _validate_vertical_segments(
    candidate_bounds: list[tuple[int, int, int, int]],
    parent_bounds: tuple[int, int, int, int],
    image_size: tuple[int, int],
    stage_label: str,
) -> tuple[list[tuple[int, int, int, int]], list[str]]:
    valid_segments: list[tuple[int, int, int, int]] = []
    child_reviews: list[str] = []
    parent_left, parent_top, parent_right, parent_bottom = parent_bounds
    parent_height = max(parent_bottom - parent_top, 1)
    parent_width = max(parent_right - parent_left, 1)
    image_area = float(max(image_size[0] * image_size[1], 1))
    parent_area = float(max((parent_right - parent_left) * (parent_bottom - parent_top), 1))

    for child_index, (seg_left_abs, seg_top_abs, seg_right_abs, seg_bottom_abs) in enumerate(candidate_bounds, start=1):
        seg_width = seg_right_abs - seg_left_abs
        seg_height = seg_bottom_abs - seg_top_abs
        child_area_ratio = (seg_width * seg_height) / image_area
        height_ratio = seg_height / max(parent_height, 1)
        parent_area_ratio = (seg_width * seg_height) / parent_area
        min_child_height = max(parent_height // 5, int(parent_height * MIN_TICKET_HEIGHT_RATIO), MIN_TICKET_CHILD_HEIGHT)
        rejection_reason = ""
        if seg_height < min_child_height:
            rejection_reason = "too short"
        elif seg_width < MIN_TICKET_CHILD_WIDTH or seg_height < MIN_TICKET_CHILD_HEIGHT:
            rejection_reason = "too small"
        elif child_area_ratio < MIN_TICKET_CHILD_AREA_RATIO:
            rejection_reason = "area ratio too small"
        elif seg_width / max(parent_width, 1) < MIN_TICKET_WIDTH_RATIO:
            rejection_reason = "too narrow"
        elif seg_height / max(seg_width, 1) > MAX_FRAGMENT_ASPECT_RATIO:
            rejection_reason = "too tall and thin"

        confidence = min(max(child_area_ratio * 2.4, 0.20), 0.96)
        if rejection_reason:
            child_reviews.append(
                f"- {stage_label.capitalize()} child {child_index}: conf={confidence:.2f}, "
                f"height_ratio={height_ratio:.2f}, area_ratio={child_area_ratio:.3f}, "
                f"parent_area_ratio={parent_area_ratio:.3f}, status=rejected, reason={rejection_reason}"
            )
            continue

        valid_segments.append((seg_left_abs, seg_top_abs, seg_right_abs, seg_bottom_abs))
        child_reviews.append(
            f"- {stage_label.capitalize()} child {child_index}: conf={confidence:.2f}, "
            f"height_ratio={height_ratio:.2f}, area_ratio={child_area_ratio:.3f}, "
            f"parent_area_ratio={parent_area_ratio:.3f}, status=accepted"
        )

    return valid_segments, child_reviews


def _should_attempt_secondary_vertical_split(
    child_bounds: tuple[int, int, int, int],
    parent_bounds: tuple[int, int, int, int],
) -> bool:
    child_left, child_top, child_right, child_bottom = child_bounds
    parent_left, parent_top, parent_right, parent_bottom = parent_bounds
    child_width = max(child_right - child_left, 1)
    child_height = max(child_bottom - child_top, 1)
    parent_height = max(parent_bottom - parent_top, 1)
    return child_height > max(int(parent_height * 0.40), 140) and child_height > child_width * 0.85


def _should_attempt_horizontal_split(
    bounds: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> bool:
    left, top, right, bottom = bounds
    width = max(right - left, 1)
    height = max(bottom - top, 1)
    image_width, _ = image_size
    return width > height * HORIZONTAL_SPLIT_WIDTH_RATIO and width > image_width * HORIZONTAL_SPLIT_IMAGE_COVERAGE


def _should_attempt_contour_pair_split(
    bounds: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> bool:
    left, top, right, bottom = bounds
    width = max(right - left, 1)
    height = max(bottom - top, 1)
    image_width, _ = image_size
    return width > height * CONTOUR_PAIR_TRIGGER_WIDTH_RATIO or width > image_width * CONTOUR_PAIR_TRIGGER_IMAGE_COVERAGE


def _split_contour_pair(
    mask: np.ndarray,
    bounds: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[list[tuple[tuple[int, int, int, int], str]], str, list[str]]:
    left, top, right, bottom = bounds
    region_mask = mask[top:bottom, left:right]
    if region_mask.size == 0:
        return [], "Contour-based pair separation was attempted but the region mask was empty.", []

    region_height, region_width = region_mask.shape
    if region_width < MIN_TICKET_CHILD_WIDTH * 2:
        return [], "Contour-based pair separation was skipped because the region was not wide enough.", []

    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    parent_area = float(max(region_width * region_height, 1))
    contour_candidates: list[tuple[int, int, int, int, float]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = (w * h) / parent_area
        if area_ratio < MIN_CONTOUR_CHILD_AREA_RATIO:
            continue
        if w < MIN_TICKET_CHILD_WIDTH or h < MIN_TICKET_CHILD_HEIGHT:
            continue

        fill_ratio = cv2.contourArea(contour) / max(w * h, 1)
        if fill_ratio < MIN_CONTOUR_FILL_RATIO:
            continue

        aspect_ratio = max(h / max(w, 1), w / max(h, 1))
        if aspect_ratio > MAX_CONTOUR_ASPECT_RATIO:
            continue

        contour_candidates.append((x, y, x + w, y + h, area_ratio))

    contour_candidates.sort(key=lambda item: (item[1], item[0], -item[4]))
    stage2_report = [
        f"- Parent region {left},{top},{right},{bottom}: contour-based pair analysis found {len(contour_candidates)} ticket-like internal contours."
    ]
    if len(contour_candidates) < 2:
        stage2_report.append("- Result: fewer than two tall internal contours were found; parent region kept.")
        return [], "Contour-based pair separation did not find two ticket-like internal contours.", stage2_report

    best_pair: tuple[tuple[int, int, int, int, float], tuple[int, int, int, int, float]] | None = None
    best_score = -1.0
    for index, left_candidate in enumerate(contour_candidates):
        for right_candidate in contour_candidates[index + 1:]:
            left_x1, left_y1, left_x2, left_y2, left_area = left_candidate
            right_x1, right_y1, right_x2, right_y2, right_area = right_candidate
            if right_x1 <= left_x1:
                continue

            left_height = left_y2 - left_y1
            right_height = right_y2 - right_y1
            height_delta = abs(left_height - right_height) / max(max(left_height, right_height), 1)
            if height_delta > MAX_CONTOUR_HEIGHT_DELTA_RATIO:
                continue

            horizontal_gap = right_x1 - left_x2
            if horizontal_gap < 0:
                continue

            overlap_height = max(0, min(left_y2, right_y2) - max(left_y1, right_y1))
            overlap_ratio = overlap_height / max(min(left_height, right_height), 1)
            if overlap_ratio < 0.45:
                continue

            pair_score = left_area + right_area + overlap_ratio - (height_delta * 0.5)
            if pair_score > best_score:
                best_score = pair_score
                best_pair = (left_candidate, right_candidate)

    if best_pair is None:
        stage2_report.append("- Result: no side-by-side contour pair passed height and overlap checks; parent region kept.")
        return [], "Contour-based pair separation found internal contours, but none formed a believable side-by-side ticket pair.", stage2_report

    left_candidate, right_candidate = best_pair
    seam_x = (left_candidate[2] + right_candidate[0]) // 2

    proposed_bounds: list[tuple[int, int, int, int]] = []
    for seg_start, seg_end in [(0, seam_x), (seam_x, region_width)]:
        seg_mask = region_mask[:, seg_start:seg_end]
        content_bounds = _content_bounds_from_mask(seg_mask)
        if content_bounds is None:
            continue
        c_left, c_top, c_right, c_bottom = content_bounds
        proposed_bounds.append(
            (
                left + seg_start + c_left,
                top + c_top,
                left + seg_start + c_right,
                top + c_bottom,
            )
        )

    valid_segments, child_reviews = _validate_horizontal_segments(proposed_bounds, bounds, image_size, "contour")
    valid_segments = _sort_bounds_by_position(valid_segments)
    valid_segments = _filter_heavily_overlapping_bounds(valid_segments)
    valid_segments = _sort_bounds_by_position(valid_segments)

    stage2_report.append(
        f"- Parent region {left},{top},{right},{bottom}: contour pair proposed {len(proposed_bounds)} child regions at split column {seam_x}."
    )
    stage2_report.extend(child_reviews)

    if len(valid_segments) < MIN_STAGE2_REGION_COUNT:
        stage2_report.append(f"- Result: only {len(valid_segments)} contour-based child regions passed validation; parent region kept.")
        return [], "Contour-based pair separation found a candidate split, but the child regions were not ticket-like enough.", stage2_report

    note = "Contour-based pair separation succeeded using two dominant side-by-side ticket contours inside a broad first-stage region."
    stage2_report.append(f"- Result: {len(valid_segments)} contour-based child regions accepted and exported.")
    return [(bounds_value, note) for bounds_value in valid_segments], "", stage2_report


def _validate_horizontal_segments(
    candidate_bounds: list[tuple[int, int, int, int]],
    parent_bounds: tuple[int, int, int, int],
    image_size: tuple[int, int],
    stage_label: str = "horizontal",
) -> tuple[list[tuple[int, int, int, int]], list[str]]:
    valid_segments: list[tuple[int, int, int, int]] = []
    child_reviews: list[str] = []
    parent_left, parent_top, parent_right, parent_bottom = parent_bounds
    parent_height = max(parent_bottom - parent_top, 1)
    parent_width = max(parent_right - parent_left, 1)
    image_area = float(max(image_size[0] * image_size[1], 1))
    parent_area = float(max((parent_right - parent_left) * (parent_bottom - parent_top), 1))

    for child_index, (seg_left_abs, seg_top_abs, seg_right_abs, seg_bottom_abs) in enumerate(candidate_bounds, start=1):
        seg_width = seg_right_abs - seg_left_abs
        seg_height = seg_bottom_abs - seg_top_abs
        area_ratio = (seg_width * seg_height) / image_area
        parent_area_ratio = (seg_width * seg_height) / parent_area
        height_ratio = seg_height / max(parent_height, 1)
        width_ratio = seg_width / max(parent_width, 1)
        rejection_reason = ""

        if seg_width < max(parent_width // 6, MIN_TICKET_CHILD_WIDTH):
            rejection_reason = "too narrow"
        elif seg_height < MIN_TICKET_CHILD_HEIGHT:
            rejection_reason = "too short"
        elif height_ratio < MIN_HORIZONTAL_CHILD_HEIGHT_RATIO:
            rejection_reason = "height ratio too small"
        elif width_ratio < MIN_HORIZONTAL_CHILD_WIDTH_RATIO:
            rejection_reason = "width ratio too small"
        elif area_ratio < MIN_TICKET_CHILD_AREA_RATIO:
            rejection_reason = "area ratio too small"
        elif seg_height / max(seg_width, 1) > MAX_FRAGMENT_ASPECT_RATIO * 1.15:
            rejection_reason = "too tall and thin"

        confidence = min(max(area_ratio * 2.4, 0.20), 0.96)
        if rejection_reason:
            child_reviews.append(
                f"- {stage_label.capitalize()} child {child_index}: conf={confidence:.2f}, height_ratio={height_ratio:.2f}, "
                f"area_ratio={area_ratio:.3f}, parent_area_ratio={parent_area_ratio:.3f}, "
                f"status=rejected, reason={rejection_reason}"
            )
            continue

        valid_segments.append((seg_left_abs, seg_top_abs, seg_right_abs, seg_bottom_abs))
        child_reviews.append(
            f"- {stage_label.capitalize()} child {child_index}: conf={confidence:.2f}, height_ratio={height_ratio:.2f}, "
            f"area_ratio={area_ratio:.3f}, parent_area_ratio={parent_area_ratio:.3f}, status=accepted"
        )

    return valid_segments, child_reviews


def _sort_bounds_by_position(bounds_list: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    return sorted(bounds_list, key=lambda bounds: (bounds[1], bounds[0], bounds[3], bounds[2]))


def _detect_slender_artifact_bounds(
    mask: np.ndarray,
    image_size: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    image_width, image_height = image_size
    image_area = float(max(image_width * image_height, 1))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounds_list: list[tuple[int, int, int, int]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = (w * h) / image_area
        if area_ratio < MIN_SLENDER_ARTIFACT_AREA_RATIO or area_ratio > MAX_SLENDER_ARTIFACT_AREA_RATIO:
            continue

        width_ratio = w / max(image_width, 1)
        height_ratio = h / max(image_height, 1)
        aspect_ratio = h / max(w, 1)
        fill_ratio = cv2.contourArea(contour) / max(w * h, 1)
        edge_distance = min(x, image_width - (x + w)) / max(image_width, 1)

        if width_ratio > MAX_SLENDER_ARTIFACT_WIDTH_RATIO:
            continue
        if height_ratio < MIN_SLENDER_ARTIFACT_HEIGHT_RATIO:
            continue
        if aspect_ratio < MIN_SLENDER_ARTIFACT_ASPECT_RATIO:
            continue
        if fill_ratio < MIN_SLENDER_ARTIFACT_FILL_RATIO:
            continue
        if edge_distance > SLENDER_ARTIFACT_EDGE_RATIO:
            continue

        bounds_list.append((x, y, x + w, y + h))

    return _sort_bounds_by_position(bounds_list)


def _sort_regions_for_output(regions: list[SplitRegion]) -> list[SplitRegion]:
    return sorted(regions, key=lambda region: (region.bounds[1], region.bounds[0], region.bounds[3], region.bounds[2]))


def _filter_heavily_overlapping_bounds(
    bounds_list: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    filtered: list[tuple[int, int, int, int]] = []
    for bounds in bounds_list:
        if any(_intersection_over_union(bounds, existing) > MAX_ACCEPTED_REGION_IOU for existing in filtered):
            continue
        filtered.append(bounds)
    return filtered


def _filter_heavily_overlapping_regions(regions: list[SplitRegion]) -> list[SplitRegion]:
    filtered: list[SplitRegion] = []
    for region in regions:
        if any(_intersection_over_union(region.bounds, existing.bounds) > MAX_ACCEPTED_REGION_IOU for existing in filtered):
            continue
        filtered.append(region)
    return filtered


def _intersection_over_union(
    bounds_a: tuple[int, int, int, int],
    bounds_b: tuple[int, int, int, int],
) -> float:
    left = max(bounds_a[0], bounds_b[0])
    top = max(bounds_a[1], bounds_b[1])
    right = min(bounds_a[2], bounds_b[2])
    bottom = min(bounds_a[3], bounds_b[3])

    if right <= left or bottom <= top:
        return 0.0

    intersection = float((right - left) * (bottom - top))
    area_a = float(max((bounds_a[2] - bounds_a[0]) * (bounds_a[3] - bounds_a[1]), 1))
    area_b = float(max((bounds_b[2] - bounds_b[0]) * (bounds_b[3] - bounds_b[1]), 1))
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _split_horizontal_row(
    mask: np.ndarray,
    bounds: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[list[tuple[tuple[int, int, int, int], str]], str, list[str]]:
    left, top, right, bottom = bounds
    region_mask = mask[top:bottom, left:right]
    if region_mask.size == 0:
        return [], "Second-stage horizontal segmentation was attempted but the region mask was empty.", []

    col_density = np.count_nonzero(region_mask > 0, axis=0) / max(region_mask.shape[0], 1)
    smoothed_density = _smooth_density(col_density, window=max(region_mask.shape[1] // 24, 9))
    gap_runs = _find_soft_gap_runs(
        smoothed_density,
        low_threshold=0.10,
        valley_threshold=0.19,
        min_run=max(region_mask.shape[1] // 34, 8),
    )
    if not gap_runs:
        return [], "Second-stage horizontal segmentation was attempted but no separator band was strong enough.", []

    candidates = []
    start = 0
    for gap_start, gap_end in gap_runs:
        midpoint = (gap_start + gap_end) // 2
        candidates.append((start, midpoint))
        start = midpoint
    candidates.append((start, region_mask.shape[1]))

    proposed_bounds: list[tuple[int, int, int, int]] = []
    for seg_start, seg_end in candidates:
        seg_mask = region_mask[:, seg_start:seg_end]
        content_bounds = _content_bounds_from_mask(seg_mask)
        if content_bounds is None:
            continue
        c_left, c_top, c_right, c_bottom = content_bounds
        proposed_bounds.append(
            (
                left + seg_start + c_left,
                top + c_top,
                left + seg_start + c_right,
                top + c_bottom,
            )
        )

    valid_segments, child_reviews = _validate_horizontal_segments(proposed_bounds, bounds, image_size)
    valid_segments = _sort_bounds_by_position(valid_segments)
    valid_segments = _filter_heavily_overlapping_bounds(valid_segments)
    valid_segments = _sort_bounds_by_position(valid_segments)

    stage2_report = [f"- Parent region {left},{top},{right},{bottom}: {len(proposed_bounds)} horizontal child regions proposed."]
    stage2_report.extend(child_reviews)

    if len(valid_segments) < MIN_STAGE2_REGION_COUNT:
        stage2_report.append(f"- Result: only {len(valid_segments)} valid horizontal child regions remained; parent region kept.")
        return [], "Second-stage horizontal segmentation was attempted but did not produce enough ticket-like child regions.", stage2_report

    note = "Second-stage horizontal segmentation succeeded using a low-density vertical gap inside a broad first-stage region."
    stage2_report.append(f"- Result: {len(valid_segments)} valid horizontal child regions accepted and exported.")
    return [(bounds_value, note) for bounds_value in valid_segments], "", stage2_report


def _find_gap_runs(density: np.ndarray, threshold: float, min_run: int) -> list[tuple[int, int]]:
    runs = []
    run_start = None
    for index, value in enumerate(density):
        if value <= threshold:
            if run_start is None:
                run_start = index
        elif run_start is not None:
            if index - run_start >= min_run:
                runs.append((run_start, index))
            run_start = None
    if run_start is not None and len(density) - run_start >= min_run:
        runs.append((run_start, len(density)))
    return runs


def _find_soft_gap_runs(
    density: np.ndarray,
    low_threshold: float,
    valley_threshold: float,
    min_run: int,
) -> list[tuple[int, int]]:
    runs = []
    run_start = None
    valley_seen = False

    for index, value in enumerate(density):
        if value <= valley_threshold:
            if run_start is None:
                run_start = index
                valley_seen = value <= low_threshold
            else:
                valley_seen = valley_seen or value <= low_threshold
        elif run_start is not None:
            if valley_seen and index - run_start >= min_run:
                runs.append((run_start, index))
            run_start = None
            valley_seen = False

    if run_start is not None and valley_seen and len(density) - run_start >= min_run:
        runs.append((run_start, len(density)))

    return runs


def _smooth_density(density: np.ndarray, window: int) -> np.ndarray:
    if len(density) == 0:
        return density
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(density, (window // 2, window // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32)


def _content_bounds_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    left = int(xs.min())
    top = int(ys.min())
    right = int(xs.max()) + 1
    bottom = int(ys.max()) + 1
    return (left, top, right, bottom)


def _merge_overlapping_bounds(bounds_list: list[tuple[int, int, int, int]], image_size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    if not bounds_list:
        return []

    merged: list[tuple[int, int, int, int]] = []
    image_width, image_height = image_size

    for bounds in bounds_list:
        current = bounds
        changed = True
        while changed:
            changed = False
            next_merged: list[tuple[int, int, int, int]] = []
            for existing in merged:
                if _should_merge(current, existing, image_width, image_height):
                    current = (
                        min(current[0], existing[0]),
                        min(current[1], existing[1]),
                        max(current[2], existing[2]),
                        max(current[3], existing[3]),
                    )
                    changed = True
                else:
                    next_merged.append(existing)
            merged = next_merged
        merged.append(current)

    merged.sort(key=lambda box: (box[1], box[0]))
    return merged


def _should_merge(a: tuple[int, int, int, int], b: tuple[int, int, int, int], image_width: int, image_height: int) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    overlap_x = max(0, min(ax2, bx2) - max(ax1, bx1))
    overlap_y = max(0, min(ay2, by2) - max(ay1, by1))

    gap_x = max(0, max(ax1, bx1) - min(ax2, bx2))
    gap_y = max(0, max(ay1, by1) - min(ay2, by2))

    horizontal_merge = overlap_y > min((ay2 - ay1), (by2 - by1)) * 0.35 and gap_x < image_width * 0.04
    vertical_merge = overlap_x > min((ax2 - ax1), (bx2 - bx1)) * 0.35 and gap_y < image_height * 0.04
    direct_overlap = overlap_x > 0 and overlap_y > 0

    return direct_overlap or horizontal_merge or vertical_merge


def _pad_bounds(bounds: tuple[int, int, int, int], image_size: tuple[int, int], pad_ratio: float, min_pad: int) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    left, top, right, bottom = bounds
    width = max(right - left, 1)
    height = max(bottom - top, 1)
    pad_x = max(int(width * pad_ratio), min_pad)
    pad_y = max(int(height * pad_ratio), min_pad)
    return (
        max(left - pad_x, 0),
        max(top - pad_y, 0),
        min(right + pad_x, image_width),
        min(bottom + pad_y, image_height),
    )
