# SPDX-License-Identifier: GPL-3.0-only
"""Generic helpers for demo.py"""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


# --- Device and video I/O --------------------------------------------------


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def video_info(path: Path) -> tuple[int, int, float, int | None]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Only an estimate (used for progress bars); the true count comes from decoding.
    frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) or None
    cap.release()
    return width, height, fps, frames


class VideoSink:
    """H.264 via ffmpeg when available (plays everywhere); mp4v otherwise."""

    def __init__(self, path, fps, width, height):
        self.proc = self.writer = None
        if shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                    "-r", f"{fps}", "-i", "-",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
                ],
                stdin=subprocess.PIPE,
            )
        else:
            print("      ffmpeg not found; writing mp4v (install ffmpeg for H.264).")
            self.writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )

    def write(self, frame):
        if self.proc:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.writer.write(frame)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            if self.proc.wait():
                raise RuntimeError("ffmpeg failed to encode the output video")
        else:
            self.writer.release()


# --- Probabilities ---------------------------------------------------------


def to_probabilities(logits):
    """Softmax for multiclass heads; [1 - p, p] for the binary sigmoid head."""
    if len(logits) == 1:
        probability = 1.0 / (1.0 + np.exp(-logits[0]))
        return np.array([1.0 - probability, probability], dtype=np.float32)
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()


def from_probabilities(probabilities, binary):
    """Inverse of to_probabilities, up to a constant offset for multiclass."""
    probabilities = np.clip(probabilities, 1e-5, 1.0 - 1e-5)
    if binary:
        return np.array([np.log(probabilities[1] / probabilities[0])], dtype=np.float32)
    return np.log(probabilities).astype(np.float32)


# --- Drawing primitives ----------------------------------------------------


@functools.lru_cache(maxsize=None)
def load_font(size, bold):
    """Helvetica Neue on macOS, else DejaVu Sans, else Pillow's default."""
    dejavu = f"DejaVuSans{'-Bold' if bold else ''}.ttf"
    candidates = [
        ("/System/Library/Fonts/HelveticaNeue.ttc", 1 if bold else 10),
        (f"/usr/share/fonts/truetype/dejavu/{dejavu}", 0),  # most Linux distros
    ]
    # matplotlib ships DejaVu; locate it without importing matplotlib, whose import
    # fails when MPLBACKEND names an unavailable backend (e.g. in Colab).
    spec = importlib.util.find_spec("matplotlib")
    if spec and spec.submodule_search_locations:
        fonts = Path(spec.submodule_search_locations[0]) / "mpl-data" / "fonts" / "ttf"
        candidates.append((fonts / dejavu, 0))
    for path, index in candidates:
        with contextlib.suppress(OSError):
            return ImageFont.truetype(str(path), size, index=index)
    return ImageFont.load_default(size)


@functools.lru_cache(maxsize=1024)
def text_mask(text, size, bold):
    """Anti-aliased coverage mask of `text`, plus the y of its cap-height center."""
    font = load_font(size, bold)
    ascent, descent = font.getmetrics()
    left, _, right, _ = font.getbbox(text)
    image = Image.new("L", (max(1, right - left + 2), ascent + descent))
    ImageDraw.Draw(image).text((1 - left, 0), text, font=font, fill=255)
    _, cap_top, _, cap_bottom = font.getbbox("H")
    return np.asarray(image, np.float32)[..., None] / 255.0, (cap_top + cap_bottom) / 2


def text_width(text, size, bold=False):
    return text_mask(text, size, bold)[0].shape[1]


def put_text(img, text, x, cy, size, color, bold=False, opacity=1.0, align="left"):
    """Blend text whose capitals are vertically centered on `cy`."""
    mask, cap_center = text_mask(text, size, bold)
    if align == "right":
        x -= mask.shape[1]
    x, y = int(round(x)), int(round(cy - cap_center))
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = max(0, x), max(0, y), min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x2 <= x1 or y2 <= y1:
        return
    alpha = mask[y1 - y:y2 - y, x1 - x:x2 - x] * opacity
    roi = img[y1:y2, x1:x2]
    roi[:] = (roi * (1.0 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)


def blend_shape(img, bounds, alpha, draw):
    """Run `draw(layer, dx, dy)` on a copy of the region and alpha-blend it back."""
    x1, y1, x2, y2 = (int(v) for v in bounds)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1 or alpha <= 0:
        return
    roi = img[y1:y2, x1:x2]
    layer = roi.copy()
    draw(layer, -x1, -y1)
    roi[:] = cv2.addWeighted(layer, alpha, roi, 1.0 - alpha, 0)


def rounded_rect(x1, y1, x2, y2, radius):
    radius = int(max(0, min(radius, (x2 - x1) / 2, (y2 - y1) / 2)))
    corners = ((x2 - radius, y1 + radius, 270), (x2 - radius, y2 - radius, 0),
               (x1 + radius, y2 - radius, 90), (x1 + radius, y1 + radius, 180))
    return np.concatenate([
        cv2.ellipse2Poly((int(cx), int(cy)), (radius, radius), 0, start, start + 90, 10)
        for cx, cy, start in corners
    ])


def fill_rounded(img, box, radius, color, alpha):
    points = rounded_rect(*box, radius)
    blend_shape(img, (box[0] - 1, box[1] - 1, box[2] + 2, box[3] + 2), alpha,
                lambda layer, dx, dy: cv2.fillPoly(layer, [points + (dx, dy)], color, cv2.LINE_AA))


def stroke_rounded(img, box, radius, color, thickness, alpha):
    points = rounded_rect(*box, radius)
    pad = thickness + 2
    blend_shape(img, (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), alpha,
                lambda layer, dx, dy: cv2.polylines(
                    layer, [points + (dx, dy)], True, color, thickness, cv2.LINE_AA))


def draw_dot(img, center, radius, color, alpha):
    x, y = center
    pad = int(radius) + 2
    blend_shape(img, (x - pad, y - pad, x + pad + 1, y + pad + 1), alpha,
                lambda layer, dx, dy: cv2.circle(
                    layer, (x + dx, y + dy), int(radius), color, -1, cv2.LINE_AA))
