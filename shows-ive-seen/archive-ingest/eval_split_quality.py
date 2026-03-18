from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import ImageDraw
from scan_splitting import load_image, split_ticket_scan


FIXTURE_DIR = Path(__file__).resolve().parent / "test-fixtures" / "multi-ticket-image"
REVIEW_DIR = Path(__file__).resolve().parent / "review" / "contact-sheets"
IOU_PASS_THRESHOLD = 0.5
MERGE_OVERLAP_THRESHOLD = 0.15


@dataclass
class Box:
    label: str
    bounds: tuple[int, int, int, int]


def load_expected_boxes() -> tuple[Path, list[Box]]:
    fixture_path = FIXTURE_DIR / "expected_boxes.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    source_path = FIXTURE_DIR / payload["source_image"]
    expected_size = (payload["image_size"]["width"], payload["image_size"]["height"])
    actual_image = load_image(source_path)
    actual_size = actual_image.size
    scale_x = actual_size[0] / max(expected_size[0], 1)
    scale_y = actual_size[1] / max(expected_size[1], 1)

    boxes = []
    for artifact in payload["artifacts"]:
        bbox = artifact["bbox"]
        x = bbox["x"]
        y = bbox["y"]
        width = bbox["width"]
        height = bbox["height"]
        if expected_size != actual_size:
            x = round(x * scale_x)
            y = round(y * scale_y)
            width = round(width * scale_x)
            height = round(height * scale_y)
        boxes.append(
            Box(
                label=artifact["label"],
                bounds=(
                    x,
                    y,
                    x + width,
                    y + height,
                ),
            )
        )
    return source_path, boxes, expected_size, actual_size


def intersection_over_union(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = float((right - left) * (bottom - top))
    area_a = float(max((a[2] - a[0]) * (a[3] - a[1]), 1))
    area_b = float(max((b[2] - b[0]) * (b[3] - b[1]), 1))
    union = area_a + area_b - intersection
    return intersection / max(union, 1.0)


def box_contains_point(bounds: tuple[int, int, int, int], point: tuple[float, float]) -> bool:
    return bounds[0] <= point[0] <= bounds[2] and bounds[1] <= point[1] <= bounds[3]


def center_of(bounds: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0)


def format_bounds(bounds: tuple[int, int, int, int]) -> str:
    return f"({bounds[0]}, {bounds[1]}, {bounds[2]}, {bounds[3]})"


def write_debug_overlay(source_path: Path, expected_boxes: list[Box], predicted_boxes: list[Box]) -> Path:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    image = load_image(source_path).copy()
    draw = ImageDraw.Draw(image)

    for box in expected_boxes:
        draw.rectangle(box.bounds, outline=(220, 60, 60), width=6)
        draw.text((box.bounds[0] + 8, max(box.bounds[1] - 24, 8)), f"E:{box.label}", fill=(220, 60, 60))

    for box in predicted_boxes:
        draw.rectangle(box.bounds, outline=(40, 110, 230), width=4)
        draw.text((box.bounds[0] + 8, min(box.bounds[1] + 8, image.height - 24)), f"P:{box.label}", fill=(40, 110, 230))

    overlay_path = REVIEW_DIR / "multi-ticket-image-eval-overlay.jpg"
    image.save(overlay_path, format="JPEG", quality=95, subsampling=0)
    return overlay_path


def evaluate() -> tuple[bool, list[str]]:
    source_path, expected_boxes, expected_size, actual_size = load_expected_boxes()
    result = split_ticket_scan(load_image(source_path), source_path.name)
    predicted_boxes = [Box(label=f"predicted_{index}", bounds=region.bounds) for index, region in enumerate(result.regions, start=1)]
    overlay_path = write_debug_overlay(source_path, expected_boxes, predicted_boxes)

    diagnostics = [
        f"[INFO] Fixture source: {source_path}",
        f"[INFO] source.jpg dimensions: {actual_size[0]}x{actual_size[1]}",
        f"[INFO] expected_boxes.json image_size: {expected_size[0]}x{expected_size[1]}",
        f"[INFO] Expected artifact count: {len(expected_boxes)}",
        f"[INFO] Predicted artifact count: {len(predicted_boxes)}",
        f"[INFO] Debug overlay: {overlay_path}",
    ]
    if expected_size != actual_size:
        diagnostics.append("[WARN] expected_boxes.json image_size differed from source.jpg and was scaled into source-image coordinates.")

    diagnostics.append("[INFO] Expected boxes:")
    for box in expected_boxes:
        diagnostics.append(f"  - {box.label}: {format_bounds(box.bounds)}")

    diagnostics.append("[INFO] Predicted boxes:")
    for box in predicted_boxes:
        diagnostics.append(f"  - {box.label}: {format_bounds(box.bounds)}")

    matched: list[tuple[str, str, float]] = []
    missed: list[str] = []
    extra: list[str] = []
    merged: list[str] = []
    used_predictions: set[int] = set()
    passed = True

    for expected in expected_boxes:
        best_index = -1
        best_iou = 0.0
        for index, predicted in enumerate(predicted_boxes):
            iou = intersection_over_union(expected.bounds, predicted.bounds)
            if iou > best_iou:
                best_iou = iou
                best_index = index

        diagnostics.append(f"[INFO] IoU for {expected.label}: {best_iou:.3f}")
        if best_index >= 0 and best_iou >= IOU_PASS_THRESHOLD:
            matched.append((expected.label, predicted_boxes[best_index].label, best_iou))
            used_predictions.add(best_index)
        else:
            missed.append(expected.label)
            passed = False

    for index, predicted in enumerate(predicted_boxes):
        if index not in used_predictions:
            extra.append(predicted.label)

        covered_expected = [
            expected.label
            for expected in expected_boxes
            if intersection_over_union(predicted.bounds, expected.bounds) >= MERGE_OVERLAP_THRESHOLD
            or box_contains_point(predicted.bounds, center_of(expected.bounds))
        ]
        if len(covered_expected) > 1:
            merged.append(f"{predicted.label} -> {', '.join(covered_expected)}")
            passed = False

    if len(predicted_boxes) != len(expected_boxes):
        passed = False
        diagnostics.append(f"[FAIL] Expected exactly {len(expected_boxes)} predicted artifacts, found {len(predicted_boxes)}.")
    else:
        diagnostics.append(f"[PASS] Predicted artifact count matches expected value ({len(expected_boxes)}).")

    if matched:
        diagnostics.append("[INFO] Matched artifacts:")
        for expected_label, predicted_label, iou in matched:
            diagnostics.append(f"  - {expected_label} matched {predicted_label} with IoU {iou:.3f}")
    else:
        diagnostics.append("[INFO] Matched artifacts: none")

    if missed:
        diagnostics.append(f"[FAIL] Missed artifacts: {', '.join(missed)}")
    else:
        diagnostics.append("[PASS] No expected artifacts were missed.")

    if extra:
        diagnostics.append(f"[FAIL] Extra artifacts: {', '.join(extra)}")
        passed = False
    else:
        diagnostics.append("[PASS] No extra artifacts were predicted.")

    if merged:
        diagnostics.append("[FAIL] Merged artifact candidates detected:")
        for item in merged:
            diagnostics.append(f"  - {item}")
    else:
        diagnostics.append("[PASS] No predicted artifact appears to contain more than one expected artifact.")

    if all(iou >= IOU_PASS_THRESHOLD for _, _, iou in matched) and len(matched) == len(expected_boxes):
        diagnostics.append(f"[PASS] All matched artifacts met the IoU threshold ({IOU_PASS_THRESHOLD:.2f}).")
    else:
        diagnostics.append(f"[FAIL] One or more expected artifacts did not meet the IoU threshold ({IOU_PASS_THRESHOLD:.2f}).")
        passed = False

    return passed, diagnostics


def main() -> int:
    passed, diagnostics = evaluate()
    for line in diagnostics:
        print(line)
    print("\nResult: PASS" if passed else "\nResult: FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
