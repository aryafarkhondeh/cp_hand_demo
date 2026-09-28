#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only
#

from __future__ import annotations

import argparse
import contextlib
import pickle
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from hiera.hiera import hiera_base_16x224
from tqdm import tqdm
from utils import (
    VideoSink, choose_device, draw_dot, fill_rounded, from_probabilities, put_text,
    stroke_rounded, text_width, to_probabilities, video_info,
)


SRC = Path(__file__).resolve().parent
ROOT = SRC.parent  # repository root; checkpoints live in ROOT / "checkpoints"
# One model tracks people and gives their 17 COCO body keypoints (elbow, wrist).
DETECTOR = "yolo26m-pose"
LABELS = {
    "object": ("non_oih", "oih"),
    "manipulation": ("background", "grasp", "hold", "operate", "release"),
}
OUTPUT_DIMS = {"object": 1, "manipulation": 5}
DEFAULT_TASK = "manipulation"
MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32)
STD = np.array([0.225, 0.225, 0.255], dtype=np.float32)
# One hue per person (BGR): teal, coral, violet, amber, sky, lime, orange, pink.
PERSON_COLORS = (
    (191, 212, 45), (133, 113, 251), (250, 139, 167), (36, 191, 251),
    (248, 189, 56), (53, 230, 163), (60, 146, 251), (182, 114, 244),
)
# Labels below this confidence are hidden; stronger ones are drawn more opaque.
MIN_CONFIDENCE = 0.4
FULL_CONFIDENCE = 0.9
# Tracks shorter than this are usually detector false positives.
MIN_TRACK_FRAMES = 15
# Hiera sees +-16 frames around the predicted one; frames closer than this to an
# end of the video would need padding, so they reuse the nearest full prediction.
EDGE = 16


class Hiera(torch.nn.Module):
    """The same Hiera backbone used by the training repository."""

    def __init__(self):
        super().__init__()
        self.encoder = hiera_base_16x224()
        self.encoder.head = nn.Identity()

    def forward(self, images):
        return self.encoder(images)


class HandHead(nn.Module):
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, OUTPUT_DIMS[task])
        )

    def forward(self, features):
        return {f"pred_{self.task}": self.classifier(self.dropout(features))}


class DemoModel(nn.Module):
    """Module names match the Lightning checkpoint (encoder.* and decoder.*)."""

    def __init__(self, task):
        super().__init__()
        self.encoder = Hiera()
        self.decoder = HandHead(task)

    def forward(self, images):
        return self.decoder(self.encoder(images))


def gost(num_frames: int, height: int, width: int) -> dict:
    return {
        "img_shape": (height, width),
        "body_bbox": np.full((num_frames, 4), -1, np.float32),
        "body_keypoint": np.zeros((num_frames, 17, 2), np.float32),
        "body_keypoint_score": np.zeros((num_frames, 17), np.float32),
        "left_hand_bbox": np.full((num_frames, 4), -1, np.float32),
        "right_hand_bbox": np.full((num_frames, 4), -1, np.float32),
    }


def hand_box(image_shape, person_box, keypoints, elbow: int, wrist: int) -> np.ndarray:
    """Create the exact pseudo hand box used by the original demo."""
    height, width = image_shape
    elbow_xy, wrist_xy = keypoints[elbow], keypoints[wrist]
    if (
        not np.isfinite(elbow_xy).all()
        or not np.isfinite(wrist_xy).all()
        or person_box[2] <= person_box[0]
        or person_box[3] <= person_box[1]
    ):
        return np.full(4, -1, np.float32)
    center = wrist_xy + 0.5 * (wrist_xy - elbow_xy)
    half_dim = 0.4 * min(person_box[2] - person_box[0], person_box[3] - person_box[1]) / 2
    x, y = center
    box = np.array(
        [max(0, x - half_dim), max(0, y - half_dim), min(width, x + half_dim), min(height, y + half_dim)],
        dtype=np.float32,
    )
    return box if box[2] > box[0] and box[3] > box[1] else np.full(4, -1, np.float32)


def extract_poses(video_path, expected_frames, height, width, device, detector):
    """Track all people and get their body keypoints in one YOLO-pose pass.

    Returns the people and the number of frames actually read: container
    metadata (expected_frames) is only an estimate for many videos.
    """
    from ultralytics import YOLO

    print("[1/3] Tracking people and extracting poses ...")
    # Ultralytics downloads the weights to this path on first use.
    weights = ROOT / "checkpoints" / f"{detector}.pt"
    weights.parent.mkdir(exist_ok=True)
    yolo = YOLO(str(weights))
    detections, num_frames = {}, 0
    stream = yolo.track(
        source=str(video_path), stream=True, tracker="botsort.yaml", persist=True,
        classes=[0], verbose=False, device=device,
    )
    for frame_idx, result in enumerate(tqdm(stream, total=expected_frames, unit="frame")):
        num_frames = frame_idx + 1
        if result.boxes.id is None:
            continue
        ids = result.boxes.id.detach().cpu().numpy().astype(int)
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        keypoints = result.keypoints.xy.detach().cpu().numpy().astype(np.float32)
        scores = result.keypoints.conf.detach().cpu().numpy().astype(np.float32)
        for pid, bbox, body_kp, body_score in zip(ids, boxes, keypoints, scores):
            detections.setdefault(int(pid), []).append((frame_idx, bbox, body_kp, body_score))
    people = {}
    for pid, rows in detections.items():
        person = people[pid] = gost(num_frames, height, width)
        for frame_idx, bbox, body_kp, body_score in rows:
            person["body_bbox"][frame_idx] = bbox
            person["body_keypoint"][frame_idx] = body_kp
            person["body_keypoint_score"][frame_idx] = body_score
            person["left_hand_bbox"][frame_idx] = hand_box(
                (height, width), bbox, body_kp, 7, 9
            )
            person["right_hand_bbox"][frame_idx] = hand_box(
                (height, width), bbox, body_kp, 8, 10
            )
    return people, num_frames


def crop_box(boxes, height, width):
    valid = boxes[(boxes >= 0).all(axis=1)]
    if not len(valid):
        return 0, 0, width, height
    x1, y1 = valid[:, :2].min(axis=0)
    x2, y2 = valid[:, 2:].max(axis=0)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return 0, 0, width, height
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    radius = max((x2 - x1) / 2, (y2 - y1) / 2)
    return (
        max(0, int(cx - radius)), max(0, int(cy - radius)),
        min(width, int(cx + radius)), min(height, int(cy + radius)),
    )


def make_clip(frame_cache, person, hand, center, num_frames, height, width):
    global_ids = np.arange(max(0, center - 16), min(center + 16, num_frames))
    # Always produce 16 temporal samples. At either video boundary, replicate
    # the nearest available frame instead of feeding black frames to Hiera.
    # This is especially important at the end, where zero-padding otherwise
    # makes the model abruptly switch to the background class.
    rgb_ids = np.clip(
        np.arange(center - 16, center + 16, 2), 0, num_frames - 1
    ).tolist()
    x1, y1, x2, y2 = crop_box(
        person[f"{hand}_hand_bbox"][global_ids], height, width
    )
    out = np.zeros((16, 224, 224, 3), dtype=np.float32)
    for slot, frame_id in enumerate(rgb_ids):
        crop = frame_cache[frame_id][y1:y2, x1:x2]
        if crop.size:
            out[slot] = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
    out = (out / 255.0 - MEAN) / STD
    return torch.from_numpy(out.transpose(3, 0, 1, 2)).contiguous()


def checkpoint_for(task):
    return ROOT / "checkpoints" / f"hiera_{task}_hand.ckpt"


def load_model(task, device):
    model = DemoModel(task)
    checkpoint = checkpoint_for(task)
    saved = torch.load(checkpoint, map_location="cpu")
    state = saved.get("state_dict", saved)
    remapped = {}
    for key, value in state.items():
        key = key.removeprefix("model.")
        key = key.replace(f"decoder.cls_{task}.", "decoder.classifier.")
        if key.startswith(("encoder.", "decoder.")):
            remapped[key] = value
    try:
        model.load_state_dict(remapped, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Checkpoint does not match Hiera-Hand: {exc}") from exc
    return model.eval().to(device)


def predict_actions(
    video_path, people, task, device, batch_size,
    num_frames, height, width, stride=1,
):
    """Decode once with a rolling cache and batch overlapping hand clips."""
    print("[2/3] Running batched hand-action recognition ...")
    model = load_model(task, device)
    cap = cv2.VideoCapture(str(video_path))
    cache, order = {}, deque()
    queued_images, queued_keys = [], []
    predictions = {pid: [dict() for _ in range(num_frames)] for pid in people}

    def flush():
        if not queued_images:
            return
        images = torch.stack(queued_images).to(device, non_blocking=True)
        amp = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.startswith("cuda") else contextlib.nullcontext()
        )
        with torch.inference_mode(), amp:
            logits = model(images)[f"pred_{task}"].float().cpu().numpy()
        for (pid, frame_id, hand), row in zip(queued_keys, logits):
            predictions[pid][frame_id][hand] = row
        queued_images.clear()
        queued_keys.clear()

    def enqueue_center(center):
        # Only frames with a complete +-16-frame window: padding the missing frames
        # near either end makes Hiera see frozen motion and flip to background.
        # fill_edges() gives edge frames the nearest complete prediction.
        first, last = EDGE, num_frames - EDGE + 1
        if first <= last and not first <= center <= last:
            return
        if center % stride and center not in (first, last):
            return
        for pid, person in people.items():
            # Track IDs are allocated for the full video, but inference is only
            # useful while that person is actually visible at the center frame.
            if (person["body_bbox"][center] < 0).any():
                continue
            for hand in ("left", "right"):
                queued_images.append(
                    make_clip(cache, person, hand, center, num_frames, height, width)
                )
                queued_keys.append((pid, center, hand))
                if len(queued_images) >= batch_size:
                    flush()

    progress = tqdm(total=num_frames, unit="frame")
    frame_id = 0
    while frame_id < num_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        cache[frame_id] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        order.append(frame_id)
        if frame_id >= 14:
            center = frame_id - 14
            enqueue_center(center)
            progress.update(1)
            while order and order[0] < center - 16:
                del cache[order.popleft()]
        frame_id += 1
    # If this decoder yields fewer frames than tracking did, stop at what exists.
    num_frames = frame_id
    for center in range(max(0, frame_id - 14), num_frames):
        enqueue_center(center)
        progress.update(1)
        while order and order[0] < center - 16:
            del cache[order.popleft()]
    flush()
    progress.close()
    cap.release()
    fill_skipped(predictions, people, stride)
    fill_edges(predictions, people, num_frames)
    return predictions


def fill_edges(predictions, people, num_frames):
    """Give the first/last frames the nearest prediction made with a complete window."""
    first, last = EDGE, num_frames - EDGE + 1
    if first > last:
        return  # video shorter than one window: every clip was padded
    for pid, frames in predictions.items():
        visible = (people[pid]["body_bbox"] >= 0).all(axis=1)
        for hand in ("left", "right"):
            known = [i for i in range(first, last + 1) if hand in frames[i]]
            if not known:
                continue
            for anchor, span in ((known[0], range(0, known[0])), (known[-1], range(known[-1] + 1, num_frames))):
                for i in span:
                    if visible[i] and hand not in frames[i] and abs(i - anchor) <= EDGE:
                        frames[i][hand] = frames[anchor][hand]


def fill_skipped(predictions, people, stride):
    """Interpolate probabilities for frames skipped by --stride.

    Gaps between two predictions at most `stride` apart are linearly
    interpolated; the ends of a visible segment reuse the nearest prediction.
    Only frames where the person is tracked are filled.
    """
    if stride <= 1:
        return
    for pid, frames in predictions.items():
        visible = (people[pid]["body_bbox"] >= 0).all(axis=1)
        for hand in ("left", "right"):
            known = [i for i, frame in enumerate(frames) if hand in frame]
            if not known:
                continue
            binary = len(frames[known[0]][hand]) == 1
            for start, end in zip(known, known[1:]):
                if end - start > stride:
                    continue
                p_start = to_probabilities(frames[start][hand])
                p_end = to_probabilities(frames[end][hand])
                for i in range(start + 1, end):
                    if visible[i]:
                        t = (i - start) / (end - start)
                        frames[i][hand] = from_probabilities((1 - t) * p_start + t * p_end, binary)
            for anchor in known:
                for i in range(max(0, anchor - stride + 1), min(len(frames), anchor + stride)):
                    if visible[i] and hand not in frames[i]:
                        frames[i][hand] = frames[anchor][hand]


def prediction_label(logits, task):
    """Return the most likely label and a calibrated display confidence."""
    if len(logits) == 1:
        probability = float(1.0 / (1.0 + np.exp(-logits[0])))
        return LABELS[task][int(probability >= 0.5)], max(probability, 1.0 - probability)
    logits = logits - np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    label_id = int(np.argmax(probabilities))
    return LABELS[task][label_id], float(probabilities[label_id])


def smoothed_prediction(predictions, pid, hand, frame_id, task, window):
    """Average probabilities over a temporal window, then return logit-like scores."""
    if window <= 1:
        return predictions[pid][frame_id].get(hand)
    radius = window // 2
    samples = []
    for sample_id in range(max(0, frame_id - radius), min(len(predictions[pid]), frame_id + radius + 1)):
        logits = predictions[pid][sample_id].get(hand)
        if logits is not None:
            samples.append(to_probabilities(logits))
    if not samples:
        return None
    binary = len(predictions[pid][frame_id].get(hand, [])) == 1
    return from_probabilities(np.mean(samples, axis=0), binary)


def displayed_label(predictions, pid, hand, frame_id, task, window):
    """The action drawn for a hand, or (None, 0.0) while it is idle."""
    logits = smoothed_prediction(predictions, pid, hand, frame_id, task, window)
    if logits is None:
        return None, 0.0
    label, confidence = prediction_label(logits, task)
    if label in ("background", "non_oih") or confidence < MIN_CONFIDENCE:
        return None, 0.0
    return label, confidence


def smoothed_bbox(person, hand, frame_id, window):
    """Average only valid nearby boxes so brief detector jitter does not jump."""
    if window <= 1:
        return person[f"{hand}_hand_bbox"][frame_id]
    radius = window // 2
    boxes = []
    all_boxes = person[f"{hand}_hand_bbox"]
    for sample_id in range(max(0, frame_id - radius), min(len(all_boxes), frame_id + radius + 1)):
        box = all_boxes[sample_id]
        if not (box < 0).any():
            boxes.append(box)
    if not boxes:
        return np.full(4, -1, dtype=np.float32)
    return np.mean(boxes, axis=0)


def drop_short_tracks(people, min_frames=MIN_TRACK_FRAMES):
    kept = {
        pid: person for pid, person in people.items()
        if (person["body_bbox"] >= 0).all(axis=1).sum() >= min_frames
    }
    if len(kept) < len(people):
        print(f"      Ignoring {len(people) - len(kept)} track(s) shorter than {min_frames} frames.")
    return kept


def confidence_opacity(confidence):
    strength = float(np.clip(
        (confidence - MIN_CONFIDENCE) / (FULL_CONFIDENCE - MIN_CONFIDENCE), 0.0, 1.0
    ))
    return 0.35 + 0.65 * strength


def hand_color(person_color, hand):
    """The person's hue for the right hand, a lighter tint of it for the left."""
    if hand == "right":
        return person_color
    return tuple(int(c + (255 - c) * 0.6) for c in person_color)


def draw_hand_box(frame, bbox, trail, color, confidence):
    """Rounded translucent box and a fading motion trail."""
    s = frame.shape[0] / 720
    opacity = confidence_opacity(confidence)
    # The wrist trail grows and brightens toward the present frame.
    for age, point in enumerate(trail, start=1):
        fade = age / len(trail)
        draw_dot(frame, point, max(1.5, 4.5 * s * fade), color, opacity * (0.15 + 0.85 * fade))
    x1, y1, x2, y2 = (int(v) for v in bbox)
    radius = round(10 * s)
    fill_rounded(frame, (x1, y1, x2, y2), radius, color, 0.12 * opacity)
    stroke_rounded(frame, (x1, y1, x2, y2), radius, color, max(1, round(2 * s)), opacity)


def draw_hand_tag(frame, bbox, color, who, label, confidence):
    """Dark pill above the box: colored dot, who ("P2 L"), action, confidence."""
    s = frame.shape[0] / 720
    opacity = confidence_opacity(confidence)
    x1, y1 = int(bbox[0]), int(bbox[1])
    size, height = round(15 * s), round(28 * s)
    pad, dot_r, gap = round(10 * s), max(2, round(4 * s)), round(7 * s)
    name, percent = label.capitalize(), f"{confidence:.0%}"
    width = (pad + 2 * dot_r + gap + text_width(who, size, True) + gap
             + text_width(name, size, True) + gap + text_width(percent, size) + pad)
    tx = int(np.clip(x1, 0, max(0, frame.shape[1] - width)))
    ty = y1 - round(6 * s) - height
    if ty < 0:
        ty = y1 + round(6 * s)
    cy = ty + height / 2
    fill_rounded(frame, (tx, ty, tx + width, ty + height), height // 2, (26, 20, 16), 0.82 * opacity)
    draw_dot(frame, (tx + pad + dot_r, int(cy)), dot_r, color, opacity)
    x = tx + pad + 2 * dot_r + gap
    put_text(frame, who, x, cy, size, color, bold=True, opacity=opacity)
    x += text_width(who, size, True) + gap
    put_text(frame, name, x, cy, size, (255, 255, 255), bold=True, opacity=opacity)
    x += text_width(name, size, True) + gap
    put_text(frame, percent, x, cy, size, (200, 190, 180), opacity=opacity)


def render(
    video_path, output_path, people, predictions, task, fps, width, height,
    smoothing_window,
):
    print("[3/3] Rendering result ...")
    cap = cv2.VideoCapture(str(video_path))
    writer = VideoSink(output_path, fps, width, height)
    # People are numbered P1, P2, ... by first appearance (not raw tracker IDs).
    first_seen = {pid: int(np.argmax((p["body_bbox"] >= 0).all(axis=1))) for pid, p in people.items()}
    numbers = {pid: i + 1 for i, pid in enumerate(sorted(people, key=first_seen.get))}
    show_people = len(people) > 1
    num_frames = len(next(iter(predictions.values())))
    frame_id = 0
    while frame_id < num_frames:
        ok, frame = cap.read()
        if not ok:
            break
        hands = []
        for pid, person in people.items():
            number = numbers[pid]
            person_color = PERSON_COLORS[(number - 1) % len(PERSON_COLORS)]
            for hand in ("left", "right"):
                color = hand_color(person_color, hand)
                who = f"P{number} {hand[0].upper()}" if show_people else hand[0].upper()
                bbox = smoothed_bbox(person, hand, frame_id, smoothing_window)
                if (bbox < 0).any():
                    continue
                # Idle hands (label None) keep a faint box and trail but no tag.
                label, confidence = displayed_label(
                    predictions, pid, hand, frame_id, task, smoothing_window
                )
                trail = []
                for trail_id in range(max(0, frame_id - 10), frame_id):
                    trail_bbox = smoothed_bbox(person, hand, trail_id, smoothing_window)
                    if not (trail_bbox < 0).any():
                        trail.append((
                            int((trail_bbox[0] + trail_bbox[2]) / 2),
                            int((trail_bbox[1] + trail_bbox[3]) / 2),
                        ))
                hands.append((confidence, bbox, trail, color, who, label))
        # Tags go above every box; the most confident hand is drawn last.
        hands.sort(key=lambda item: item[0])
        for confidence, bbox, trail, color, who, label in hands:
            draw_hand_box(frame, bbox, trail, color, confidence)
        for confidence, bbox, trail, color, who, label in hands:
            if label:
                draw_hand_tag(frame, bbox, color, who, label, confidence)
        writer.write(frame)
        frame_id += 1
    cap.release()
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("demo_output.mp4"))
    parser.add_argument("--task", choices=LABELS, default=DEFAULT_TASK)
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, or mps")
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Hiera clips per forward pass (default: 16 on CUDA, 4 on MPS, 2 on CPU).",
    )
    parser.add_argument(
        "--smoothing-window", type=int, default=5,
        help="Temporal window for bbox/label smoothing; 0 disables it (default: 5).",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Run Hiera every N frames and interpolate between them; 1 runs every frame (default: 1).",
    )
    parser.add_argument("--save-intermediates", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    checkpoint = checkpoint_for(args.task)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"{checkpoint} not found. Download it from "
            "https://zenodo.org/records/14958923 into checkpoints/."
        )
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.smoothing_window < 0:
        raise ValueError("--smoothing-window must be non-negative")
    if args.stride < 1:
        raise ValueError("--stride must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    if args.batch_size is None:
        args.batch_size = {"cuda": 16, "mps": 4}.get(device.split(":")[0], 2)
    print(f"Device: {device}, Hiera batch size: {args.batch_size}")
    torch.set_float32_matmul_precision("medium")
    width, height, fps, expected_frames = video_info(args.video)
    timings = {}
    start = time.perf_counter()
    people, num_frames = extract_poses(
        args.video, expected_frames, height, width, device, DETECTOR
    )
    people = drop_short_tracks(people, min(MIN_TRACK_FRAMES, num_frames))
    if not people:
        raise RuntimeError("No people were tracked in the video")
    timings["track + pose"] = time.perf_counter() - start
    start = time.perf_counter()
    predictions = predict_actions(
        args.video, people, args.task, device, args.batch_size,
        num_frames, height, width, args.stride,
    )
    timings["actions"] = time.perf_counter() - start
    start = time.perf_counter()
    render(
        args.video, args.output, people, predictions, args.task, fps, width, height,
        args.smoothing_window,
    )
    timings["render"] = time.perf_counter() - start
    total = sum(timings.values())
    print(f"Timing ({num_frames} frames, {total:.1f}s total, {num_frames / total:.1f} fps):")
    for stage, seconds in timings.items():
        print(f"  {stage:<13}{seconds:7.1f}s  {num_frames / seconds:6.1f} fps  {seconds / total:4.0%}")
    if args.save_intermediates:
        with (args.output.parent / "pose.pkl").open("wb") as handle:
            pickle.dump({"video": people}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with (args.output.parent / "pred.pkl").open("wb") as handle:
            pickle.dump({"video": predictions}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
