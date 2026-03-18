from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class CleanedTicketResult:
    display_image: Image.Image
    ocr_gray_image: Image.Image
    ocr_binary_image: Image.Image
    notes: list[str]


def list_supported_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS],
        key=lambda path: path.name.lower(),
    )


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_cv(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def clean_ticket_image(image: Image.Image) -> CleanedTicketResult:
    cv_image = pil_to_cv(image)
    lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    notes = [
        "Applied illumination normalization to reduce uneven scan lighting and faded paper cast.",
        "Raised bright low-saturation paper regions toward off-white while preserving artwork and darker ink regions.",
        "Applied mild edge-preserving sharpening for OCR readability without redrawing text.",
    ]

    background = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=31, sigmaY=31)
    normalized_l = cv2.divide(l_channel, background, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(normalized_l)

    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)

    paper_mask = cv2.inRange(s_channel, 0, 52)
    bright_mask = cv2.inRange(v_channel, 118, 255)
    combined_mask = cv2.bitwise_and(paper_mask, bright_mask)
    combined_mask = cv2.GaussianBlur(combined_mask, (0, 0), sigmaX=5, sigmaY=5)

    lifted_v = cv2.addWeighted(v_channel, 0.72, np.full_like(v_channel, 255), 0.28, 0)
    blended_v = np.where(combined_mask > 0, lifted_v, v_channel).astype(np.uint8)
    whitened_hsv = cv2.merge((h_channel, s_channel, blended_v))
    whitened_bgr = cv2.cvtColor(whitened_hsv, cv2.COLOR_HSV2BGR)

    denoised = cv2.bilateralFilter(whitened_bgr, d=5, sigmaColor=20, sigmaSpace=20)
    sharp = unsharp_mask(denoised)

    gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    ocr_gray = cv2.GaussianBlur(gray, (3, 3), 0)
    ocr_binary = cv2.adaptiveThreshold(
        ocr_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )

    return CleanedTicketResult(
        display_image=cv_to_pil(sharp),
        ocr_gray_image=Image.fromarray(ocr_gray),
        ocr_binary_image=Image.fromarray(ocr_binary),
        notes=notes,
    )


def build_contact_sheet(original: Image.Image, cleaned: Image.Image, title: str) -> Image.Image:
    canvas = Image.new("RGB", (1400, 900), (245, 244, 240))
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 26), title, fill=(25, 25, 25))

    placements = [
        ("Final Archival Image (Original Upload)", original, (40, 90, 660, 840)),
        ("OCR Working Image Only (Not Final)", cleaned, (740, 90, 1360, 840)),
    ]

    for label, image, bounds in placements:
        x1, y1, x2, y2 = bounds
        draw.text((x1, y1 - 28), label, fill=(40, 40, 40))
        draw.rectangle(bounds, outline=(210, 208, 202), width=3)
        fitted = image.copy()
        fitted.thumbnail((x2 - x1 - 24, y2 - y1 - 24), Image.Resampling.LANCZOS)
        paste_x = x1 + ((x2 - x1) - fitted.width) // 2
        paste_y = y1 + ((y2 - y1) - fitted.height) // 2
        canvas.paste(fitted, (paste_x, paste_y))

    return canvas


def save_jpeg(image: Image.Image, path: Path, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=quality, subsampling=0)


def unsharp_mask(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2, sigmaY=1.2)
    return cv2.addWeighted(image, 1.18, blurred, -0.18, 0)
