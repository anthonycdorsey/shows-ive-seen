from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SHARE_SIZE = (1200, 630)
ARCHIVE_BACKGROUND = (250, 250, 248)
SHARE_BACKGROUND = (244, 241, 235)
PANEL_BACKGROUND = (253, 252, 248)
PANEL_SHADOW = (222, 217, 209)


@dataclass
class TicketDetectionResult:
    contour: np.ndarray | None
    box: np.ndarray | None
    angle: float
    confidence: float
    method: str
    image_width: int
    image_height: int
    fallback_used: bool
    content_box: tuple[int, int, int, int]


@dataclass
class ProcessedTicketResult:
    archive_image: Image.Image
    share_image: Image.Image
    preview_image: Image.Image
    detection: TicketDetectionResult
    crop_box: tuple[int, int, int, int]
    padded_crop_box: tuple[int, int, int, int]
    crop_padding: tuple[int, int]
    rotation_applied: float
    rotation_reason: str
    enhancement_summary: str
    framing_summary: str
    share_summary: str


@dataclass
class ContactSheetInputs:
    original: Image.Image
    archive_candidate: Image.Image
    share_candidate: Image.Image
    preview_candidate: Image.Image
    title: str
    subtitle: str


def list_supported_images(folder: Path) -> list[Path]:
    return sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS],
        key=lambda path: path.name.lower(),
    )


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_cv(image: Image.Image) -> np.ndarray:
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv_to_pil(image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def detect_ticket_bounds(image: Image.Image) -> TicketDetectionResult:
    cv_image = pil_to_cv(image)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edge_mask = _build_edge_mask(blurred)
    threshold_mask = _build_threshold_mask(blurred)
    combined_mask = cv2.bitwise_or(edge_mask, threshold_mask)

    image_height, image_width = gray.shape
    image_area = float(image_width * image_height)
    image_center = np.array([image_width / 2.0, image_height / 2.0])

    best_contour = None
    best_box = None
    best_angle = 0.0
    best_score = 0.0
    best_method = "fallback_content_box"

    masks = [
        (combined_mask, "combined_mask_min_area_rect"),
        (threshold_mask, "threshold_mask_min_area_rect"),
        (edge_mask, "edge_mask_min_area_rect"),
    ]

    for mask, method_name in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image_area * 0.035:
                continue

            rect = cv2.minAreaRect(contour)
            (_, _), (width, height), angle = rect
            if width < 40 or height < 40:
                continue

            box = cv2.boxPoints(rect)
            box = np.intp(box)

            area_ratio = area / image_area
            rect_area = max(width * height, 1.0)
            fill_ratio = area / rect_area
            aspect_ratio = max(width, height) / max(min(width, height), 1.0)
            center = np.array(rect[0])
            center_distance = np.linalg.norm(center - image_center)
            max_distance = np.linalg.norm(image_center)
            center_score = 1.0 - min(center_distance / max(max_distance, 1.0), 1.0)
            polygon = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
            polygon_bonus = 0.08 if 4 <= len(polygon) <= 8 else 0.0
            bounds = _box_to_bounds(box, image_width, image_height)
            bounds_width = max(bounds[2] - bounds[0], 1)
            bounds_height = max(bounds[3] - bounds[1], 1)
            footprint_score = min(bounds_width / image_width, bounds_height / image_height)

            aspect_score = 0.0
            if 1.1 <= aspect_ratio <= 4.8:
                aspect_score = 1.0
            elif 0.9 <= aspect_ratio <= 5.8:
                aspect_score = 0.55

            narrow_penalty = 0.0
            if aspect_ratio > 4.8:
                narrow_penalty += min((aspect_ratio - 4.8) * 0.10, 0.35)
            if footprint_score < 0.18:
                narrow_penalty += (0.18 - footprint_score) * 0.8

            score = (
                (area_ratio * 0.34)
                + (fill_ratio * 0.24)
                + (center_score * 0.14)
                + (aspect_score * 0.10)
                + (footprint_score * 0.26)
                + polygon_bonus
                - narrow_penalty
            )

            if score > best_score:
                best_score = score
                best_contour = contour
                best_box = box
                best_angle = _normalize_rect_angle(angle, width, height)
                best_method = method_name

    if best_box is not None:
        content_box = _box_to_bounds(best_box, image_width, image_height)
        confidence = min(best_score, 0.99)
        return TicketDetectionResult(
            contour=best_contour,
            box=best_box,
            angle=best_angle,
            confidence=confidence,
            method=best_method,
            image_width=image_width,
            image_height=image_height,
            fallback_used=False,
            content_box=content_box,
        )

    content_box = _content_box_from_mask(threshold_mask, image_width, image_height)
    content_area = max((content_box[2] - content_box[0]) * (content_box[3] - content_box[1]), 1)
    confidence = min(content_area / image_area * 0.85, 0.45)
    return TicketDetectionResult(
        contour=None,
        box=None,
        angle=0.0,
        confidence=confidence,
        method="fallback_content_box",
        image_width=image_width,
        image_height=image_height,
        fallback_used=True,
        content_box=content_box,
    )


def process_ticket_image(image: Image.Image) -> ProcessedTicketResult:
    detection = detect_ticket_bounds(image)
    rotated_image, rotation_applied, rotation_reason = auto_straighten_image(image, detection)
    rotated_detection = detect_ticket_bounds(rotated_image)

    crop_box, padded_crop_box, crop_padding = choose_crop_boxes(rotated_image, rotated_detection)
    cropped = rotated_image.crop(padded_crop_box)
    enhanced = apply_subtle_enhancement(cropped)

    archive_image = build_archive_candidate(enhanced)
    share_image = build_share_candidate(enhanced)
    preview_image = draw_preview_overlay(rotated_image, rotated_detection, padded_crop_box)

    framing_summary = (
        "Used the detected ticket bounds as the framing anchor, then applied a conservative padded crop to reduce empty background while keeping edge texture and torn borders visible."
    )
    share_summary = (
        "Scaled the ticket to sit larger in the 1200x630 frame with balanced margins and a subtle editorial panel so the image reads more like a finished share artifact than a raw scan on a blank canvas."
    )
    enhancement_summary = (
        "Applied local contrast recovery and mild bilateral smoothing to improve faded print while preserving existing paper texture and imperfections."
    )

    return ProcessedTicketResult(
        archive_image=archive_image,
        share_image=share_image,
        preview_image=preview_image,
        detection=rotated_detection,
        crop_box=crop_box,
        padded_crop_box=padded_crop_box,
        crop_padding=crop_padding,
        rotation_applied=rotation_applied,
        rotation_reason=rotation_reason,
        enhancement_summary=enhancement_summary,
        framing_summary=framing_summary,
        share_summary=share_summary,
    )


def auto_straighten_image(image: Image.Image, detection: TicketDetectionResult) -> tuple[Image.Image, float, str]:
    angle = detection.angle
    if detection.box is None and detection.fallback_used:
        return image.copy(), 0.0, "Rotation skipped because ticket bounds fell back to the content-box detector."
    if abs(angle) < 0.45:
        return image.copy(), 0.0, "Rotation skipped because the detected angle was already close enough to level."
    if detection.confidence < 0.08:
        return image.copy(), 0.0, "Rotation skipped because contour confidence was still too weak for safe straightening."

    limited_angle = max(min(angle, 18.0), -18.0)
    rotated = image.rotate(-limited_angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=ARCHIVE_BACKGROUND)
    return rotated, limited_angle, f"Applied conservative straightening using the detected contour angle ({limited_angle:.2f} degrees)."


def choose_crop_boxes(
    image: Image.Image,
    detection: TicketDetectionResult,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int]]:
    width, height = image.size

    if detection.box is not None:
        crop_box = _box_to_bounds(detection.box, width, height)
    else:
        crop_box = detection.content_box

    left, top, right, bottom = crop_box
    crop_width = max(right - left, 1)
    crop_height = max(bottom - top, 1)

    if detection.fallback_used:
        pad_ratio_x = 0.08
        pad_ratio_y = 0.08
    elif detection.confidence >= 0.42:
        pad_ratio_x = 0.045
        pad_ratio_y = 0.05
    elif detection.confidence >= 0.24:
        pad_ratio_x = 0.055
        pad_ratio_y = 0.06
    else:
        pad_ratio_x = 0.07
        pad_ratio_y = 0.07

    pad_x = max(min(int(crop_width * pad_ratio_x), 42), 16)
    pad_y = max(min(int(crop_height * pad_ratio_y), 42), 16)

    padded = (
        max(left - pad_x, 0),
        max(top - pad_y, 0),
        min(right + pad_x, width),
        min(bottom + pad_y, height),
    )
    return crop_box, padded, (pad_x, pad_y)


def apply_subtle_enhancement(image: Image.Image) -> Image.Image:
    cv_image = pil_to_cv(image)
    lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=22, sigmaSpace=22)

    return cv_to_pil(denoised)


def build_archive_candidate(image: Image.Image) -> Image.Image:
    # Keep the archive candidate close to a publishable object: just the padded crop,
    # with no extra decorative framing beyond the preserved scan background.
    background = Image.new("RGB", image.size, ARCHIVE_BACKGROUND)
    background.paste(image, (0, 0))
    return background


def build_share_candidate(image: Image.Image) -> Image.Image:
    canvas = Image.new("RGB", SHARE_SIZE, SHARE_BACKGROUND)
    shadow = ImageDraw.Draw(canvas)

    fitted = image.copy()
    fitted.thumbnail((1060, 540), Image.Resampling.LANCZOS)

    panel_margin_x = 34
    panel_margin_y = 26
    panel_bounds = (
        (SHARE_SIZE[0] - (fitted.width + panel_margin_x * 2)) // 2,
        (SHARE_SIZE[1] - (fitted.height + panel_margin_y * 2)) // 2,
    )
    panel_x = panel_bounds[0]
    panel_y = panel_bounds[1]
    panel_right = panel_x + fitted.width + panel_margin_x * 2
    panel_bottom = panel_y + fitted.height + panel_margin_y * 2

    shadow.rounded_rectangle(
        (panel_x + 8, panel_y + 10, panel_right + 8, panel_bottom + 10),
        radius=28,
        fill=PANEL_SHADOW,
    )
    shadow.rounded_rectangle(
        (panel_x, panel_y, panel_right, panel_bottom),
        radius=28,
        fill=PANEL_BACKGROUND,
    )

    paste_x = panel_x + panel_margin_x
    paste_y = panel_y + panel_margin_y
    canvas.paste(fitted, (paste_x, paste_y))
    return canvas


def draw_preview_overlay(image: Image.Image, detection: TicketDetectionResult, padded_crop_box: tuple[int, int, int, int]) -> Image.Image:
    preview = image.copy()
    draw = ImageDraw.Draw(preview)

    if detection.box is not None:
        polygon = [tuple(point) for point in detection.box.tolist()]
        draw.line(polygon + [polygon[0]], fill=(214, 51, 70), width=5)
    else:
        draw.rectangle(detection.content_box, outline=(214, 51, 70), width=5)

    draw.rectangle(padded_crop_box, outline=(33, 111, 238), width=5)
    return preview


def build_contact_sheet(inputs: ContactSheetInputs) -> Image.Image:
    canvas_width = 1600
    canvas_height = 1200
    canvas = Image.new("RGB", (canvas_width, canvas_height), (245, 244, 240))
    draw = ImageDraw.Draw(canvas)

    draw.text((40, 30), inputs.title, fill=(20, 20, 20))
    draw.text((40, 60), inputs.subtitle, fill=(90, 90, 90))

    frames = [
        (inputs.original, "Original", (40, 120, 740, 500)),
        (inputs.preview_candidate, "Detection Preview", (820, 120, 1520, 500)),
        (inputs.archive_candidate, "Archive Candidate", (40, 620, 740, 1100)),
        (inputs.share_candidate, "Share Candidate", (820, 620, 1520, 1100)),
    ]

    for frame_image, label, bounds in frames:
        x1, y1, x2, y2 = bounds
        draw.rectangle(bounds, outline=(210, 210, 205), width=3)
        draw.text((x1, y1 - 28), label, fill=(30, 30, 30))
        fitted = fit_image_within(frame_image, (x2 - x1 - 24, y2 - y1 - 24))
        paste_x = x1 + ((x2 - x1) - fitted.width) // 2
        paste_y = y1 + ((y2 - y1) - fitted.height) // 2
        canvas.paste(fitted, (paste_x, paste_y))

    return canvas


def fit_image_within(image: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail(max_size, Image.Resampling.LANCZOS)
    return fitted


def save_image(image: Image.Image, path: Path, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=quality, subsampling=0)


def build_processing_notes(
    source_path: Path,
    archive_output: Path,
    share_output: Path,
    preview_output: Path,
    contact_sheet_output: Path,
    result: ProcessedTicketResult,
) -> str:
    left, top, right, bottom = result.crop_box
    padded_left, padded_top, padded_right, padded_bottom = result.padded_crop_box
    pad_x, pad_y = result.crop_padding
    fallback_text = "Yes" if result.detection.fallback_used else "No"
    return f"""SHOWS I SAW - IMAGE PROCESSING REVIEW

Selected source image:
{source_path}

Generated review outputs:
- Archive candidate: {archive_output}
- Share candidate: {share_output}
- Detection preview: {preview_output}
- Contact sheet: {contact_sheet_output}

Detection summary:
- Method: {result.detection.method}
- Confidence: {result.detection.confidence:.2f}
- Fallback used: {fallback_text}
- Rotation applied: {result.rotation_applied:.2f} degrees
- Rotation decision: {result.rotation_reason}

Crop summary:
- Raw detected crop: left={left}, top={top}, right={right}, bottom={bottom}
- Padding applied: x={pad_x}px, y={pad_y}px
- Padded crop: left={padded_left}, top={padded_top}, right={padded_right}, bottom={padded_bottom}
- Framing rationale: {result.framing_summary}

Share framing summary:
- {result.share_summary}

Enhancement summary:
- {result.enhancement_summary}

Manual review checklist:
- Confirm the crop preserves padding around the ticket.
- Confirm torn edges and paper texture still read as authentic.
- Confirm no text appears redrawn or invented.
- Confirm the share image framing feels editorially balanced.
- Copy nothing into the live site until you approve the outputs manually.
"""


def reserve_output_path(path: Path) -> Path:
    if not path.exists():
        return path

    index = 2
    while True:
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _build_edge_mask(gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray, 35, 125)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    horizontal = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8), iterations=1)
    vertical = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8), iterations=1)
    return vertical


def _build_threshold_mask(gray: np.ndarray) -> np.ndarray:
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    opened = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    return closed


def _content_box_from_mask(mask: np.ndarray, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        margin_x = max(int(image_width * 0.03), 12)
        margin_y = max(int(image_height * 0.03), 12)
        return (margin_x, margin_y, image_width - margin_x, image_height - margin_y)

    left = max(int(xs.min()), 0)
    top = max(int(ys.min()), 0)
    right = min(int(xs.max()), image_width)
    bottom = min(int(ys.max()), image_height)
    return (left, top, right, bottom)


def _box_to_bounds(box: np.ndarray, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    xs = box[:, 0]
    ys = box[:, 1]
    left = max(int(xs.min()), 0)
    top = max(int(ys.min()), 0)
    right = min(int(xs.max()), image_width)
    bottom = min(int(ys.max()), image_height)
    return (left, top, right, bottom)


def _normalize_rect_angle(angle: float, width: float, height: float) -> float:
    if width < height:
        angle += 90.0
    while angle > 45.0:
        angle -= 90.0
    while angle < -45.0:
        angle += 90.0
    return angle

