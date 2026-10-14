"""Smart Study Monitor: webcam-based study-session reminder.

Signals used:
- Eye Aspect Ratio (EAR) from cvzone/MediaPipe face landmarks.
- COCO phone detections from a pretrained Ultralytics YOLO model.
- Time since a face was last visible.

This is a focus aid, not a medical or attention-diagnosis system.
"""
from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path
from statistics import median
from typing import Iterable

import cv2
import numpy as np
import pygame
from cvzone.FaceMeshModule import FaceMeshDetector
from ultralytics import YOLO

WINDOW_NAME = "Smart Study Monitor"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# MediaPipe Face Mesh landmark indices: outer eye corners, upper eyelid points,
# and lower eyelid points. EAR is averaged across both eyes.
# MediaPipe names are anatomical: 362/263 are its left eye and 33/133 its right.
LEFT_EYE = (362, 385, 387, 263, 373, 380)
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
EYE_LANDMARKS = sorted(set(LEFT_EYE + RIGHT_EYE))
PHONE_NAMES = {"cell phone", "mobile phone", "phone"}


def point_distance(a: Iterable[int], b: Iterable[int]) -> float:
    ax, ay = a
    bx, by = b
    return math.hypot(bx - ax, by - ay)


def eye_aspect_ratio(face: list[list[int]], ids: tuple[int, ...]) -> float | None:
    """Return the six-point EAR for one eye, or None for invalid landmarks."""
    if len(face) <= max(ids):
        return None
    p1, p2, p3, p4, p5, p6 = (face[index] for index in ids)
    horizontal = point_distance(p1, p4)
    if horizontal < 1.0:
        return None
    vertical_1 = point_distance(p2, p6)
    vertical_2 = point_distance(p3, p5)
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def face_eye_ratio(face: list[list[int]]) -> float | None:
    values = [eye_aspect_ratio(face, LEFT_EYE), eye_aspect_ratio(face, RIGHT_EYE)]
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def draw_eye_measurements(frame, face: list[list[int]]) -> None:
    """Draw the exact six landmarks used for EAR so a bad fit is visible."""
    for ids, color in ((LEFT_EYE, (0, 220, 0)), (RIGHT_EYE, (255, 180, 0))):
        if len(face) <= max(ids):
            continue
        p1, p2, p3, p4, p5, p6 = (
            tuple(int(round(value)) for value in face[index][:2]) for index in ids
        )
        cv2.line(frame, p1, p4, color, 1, cv2.LINE_AA)
        cv2.line(frame, p2, p6, (0, 80, 255), 1, cv2.LINE_AA)
        cv2.line(frame, p3, p5, (0, 80, 255), 1, cv2.LINE_AA)
        for point in (p1, p2, p3, p4, p5, p6):
            cv2.circle(frame, point, 3, color, -1, cv2.LINE_AA)


def draw_eye_status_panel(
    frame,
    face_found: bool,
    ratio: float | None,
    left_ratio: float | None,
    right_ratio: float | None,
    threshold: float,
    closed_elapsed: float,
    required_seconds: float,
) -> None:
    """Show a plain-language eye state and how close it is to an alert."""
    x, y = 16, 94
    width = min(460, frame.shape[1] - 32)
    height = 112
    cv2.rectangle(frame, (x, y), (x + width, y + height), (28, 28, 28), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (90, 90, 90), 1)

    if not face_found:
        label, color = "EYE STATUS: FACE NOT FOUND", (0, 170, 255)
    elif ratio is None:
        label, color = "EYE STATUS: MEASUREMENT UNAVAILABLE", (0, 170, 255)
    elif ratio < threshold and closed_elapsed >= required_seconds:
        label, color = "EYE STATUS: CLOSED - ALERT", (40, 40, 255)
    elif ratio < threshold:
        label, color = "EYE STATUS: CLOSING", (0, 180, 255)
    else:
        label, color = "EYE STATUS: OPEN", (70, 220, 90)
    cv2.putText(frame, label, (x + 12, y + 25), FONT, 0.52, color, 2, cv2.LINE_AA)

    if ratio is None:
        metrics = "EAR --   |   threshold {:.2f}".format(threshold)
    else:
        left_text = f"{left_ratio:.2f}" if left_ratio is not None else "--"
        right_text = f"{right_ratio:.2f}" if right_ratio is not None else "--"
        metrics = f"EAR {ratio:.2f}   |   L {left_text}  R {right_text}   |   limit {threshold:.2f}"
    cv2.putText(frame, metrics, (x + 12, y + 49), FONT, 0.43, (240, 240, 240), 1, cv2.LINE_AA)

    bar_x, bar_y = x + 12, y + 61
    bar_width, bar_height = width - 24, 13
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (72, 72, 72), -1)
    progress = min(1.0, max(0.0, closed_elapsed / required_seconds))
    if progress > 0:
        fill_color = (0, 180, 255) if progress < 1.0 else (40, 40, 255)
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + int(bar_width * progress), bar_y + bar_height),
            fill_color,
            -1,
        )
    cv2.putText(
        frame,
        f"Below limit: {closed_elapsed:.1f}s / {required_seconds:.1f}s",
        (x + 12, y + 98),
        FONT,
        0.42,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )


def phone_class_ids(names: dict | list) -> list[int]:
    """Find phone class IDs by name instead of assuming a fixed COCO index."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    ids = [int(class_id) for class_id, name in items if str(name).strip().lower() in PHONE_NAMES]
    if not ids:
        raise RuntimeError(
            "The selected YOLO model has no 'cell phone' class. "
            "Use a COCO-pretrained detection model such as yolo26n.pt."
        )
    return ids


def create_beep(enabled: bool):
    """Create a short local beep; return None when audio is unavailable."""
    if not enabled:
        return None
    try:
        pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
        sample_rate = 22050
        seconds = 0.16
        samples = np.arange(int(sample_rate * seconds), dtype=np.float32)
        wave = (np.sin(2.0 * np.pi * 880.0 * samples / sample_rate) * 6000).astype("<i2")
        return pygame.mixer.Sound(buffer=wave.tobytes())
    except (pygame.error, RuntimeError) as error:
        print(f"Audio is unavailable; visual alerts will still work ({error}).")
        return None


def load_alert_sound(path: Path, fallback):
    """Load a user-provided voice clip, falling back to the short beep if needed."""
    try:
        sound = pygame.mixer.Sound(str(path))
        sound.set_volume(1.0)
        print(f"Loaded alert voice: {path.name} ({sound.get_length():.1f}s).")
        return sound
    except (pygame.error, OSError) as error:
        print(f"Could not load alert voice {path.name}; using the beep instead ({error}).")
        return fallback


def draw_banner(frame, text: str, color: tuple[int, int, int], x: int = 18, y: int = 18):
    scale = 0.72
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    frame_width = frame.shape[1]
    x = max(8, min(x, frame_width - text_width - 28))
    top = max(text_height + 16, y)
    cv2.rectangle(frame, (x, top - text_height - 12),
                  (min(frame_width - 8, x + text_width + 20), top + baseline + 5), color, -1)
    cv2.putText(frame, text, (x + 10, top), FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_phone_boxes(frame, boxes: list[tuple[int, int, int, int, float]]):
    for x1, y1, x2, y2, confidence in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 220), 2)
        label = f"Phone detected: {confidence * 100:.0f}%"
        (width, height), baseline = cv2.getTextSize(label, FONT, 0.55, 2)
        label_y = max(height + 8, y1)
        cv2.rectangle(frame, (x1, label_y - height - 8),
                      (x1 + width + 10, label_y + baseline), (255, 0, 220), -1)
        cv2.putText(frame, label, (x1 + 5, label_y - 3), FONT, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time study reminders from your webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0).")
    parser.add_argument("--model", default="yolo26n.pt", help="Ultralytics detection model with a phone class.")
    parser.add_argument("--phone-confidence", type=float, default=0.45,
                        help="Minimum YOLO phone confidence from 0 to 1.")
    parser.add_argument("--phone-interval", type=float, default=0.35,
                        help="Seconds between phone-model inference passes.")
    parser.add_argument("--eye-threshold", type=float, default=0.20,
                        help="Closed-eye EAR threshold; default is 0.20.")
    parser.add_argument("--closed-seconds", type=float, default=0.9,
                        help="How long eyes must stay below threshold before an alert.")
    parser.add_argument("--face-lost-seconds", type=float, default=1.5,
                        help="How long the face must be missing before an alert.")
    parser.add_argument("--no-sound", action="store_true", help="Disable audio beeps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.phone_confidence < 1.0:
        raise SystemExit("--phone-confidence must be between 0 and 1.")
    if args.phone_interval <= 0 or args.closed_seconds <= 0 or args.face_lost_seconds <= 0:
        raise SystemExit("Intervals and durations must be greater than zero.")

    print(f"Loading {args.model}. The first run may download model weights.")
    model = YOLO(args.model)
    phone_ids = phone_class_ids(model.names)
    face_detector = FaceMeshDetector(
        staticMode=False, maxFaces=1, minDetectionCon=0.55, minTrackCon=0.55
    )
    beep = create_beep(not args.no_sound)
    asset_dir = Path(__file__).resolve().parent / "assets"
    sleepy_sound = load_alert_sound(asset_dir / "eye-closed.mpeg", beep) if beep else None
    phone_sound = load_alert_sound(asset_dir / "phone-detected.mpeg", beep) if beep else None

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not camera.isOpened():
        raise SystemExit(
            f"Could not open camera {args.camera}. Close other camera apps, check Windows camera privacy, "
            "or try --camera 1."
        )

    print("Camera started. Press Q to quit, C to calibrate with eyes open, [ or ] to tune the threshold.")
    session_start = time.monotonic()
    previous_frame_time = session_start
    next_phone_check = session_start
    eyes_closed_since: float | None = None
    face_missing_since: float | None = None
    left_eye_history: deque[float] = deque(maxlen=5)
    right_eye_history: deque[float] = deque(maxlen=5)
    show_eye_debug = False
    cached_phone_boxes: list[tuple[int, int, int, int, float]] = []
    eye_threshold = args.eye_threshold
    last_audio_time = 0.0
    active_audio_alert: str | None = None
    audio_channel = None
    phone_alert_seconds = 0.0
    sleepy_alert_seconds = 0.0
    face_missing_alert_seconds = 0.0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera stopped returning frames.")
                break

            frame = cv2.flip(frame, 1)
            now = time.monotonic()
            delta = min(max(now - previous_frame_time, 0.0), 0.25)
            previous_frame_time = now

            _, faces = face_detector.findFaceMesh(frame, draw=False)
            face_found = bool(faces)
            left_ratio = right_ratio = ratio = None
            if face_found:
                left_value = eye_aspect_ratio(faces[0], LEFT_EYE)
                right_value = eye_aspect_ratio(faces[0], RIGHT_EYE)
                if left_value is None:
                    left_eye_history.clear()
                else:
                    left_eye_history.append(left_value)
                    left_ratio = median(left_eye_history)
                if right_value is None:
                    right_eye_history.clear()
                else:
                    right_eye_history.append(right_value)
                    right_ratio = median(right_eye_history)
                valid_eye_ratios = [value for value in (left_ratio, right_ratio) if value is not None]
                ratio = sum(valid_eye_ratios) / len(valid_eye_ratios) if valid_eye_ratios else None
            else:
                left_eye_history.clear()
                right_eye_history.clear()

            if face_found:
                face_missing_since = None
            elif face_missing_since is None:
                face_missing_since = now
            face_missing = (
                face_missing_since is not None
                and now - face_missing_since >= args.face_lost_seconds
            )

            sleepy = False
            if ratio is not None:
                if ratio < eye_threshold:
                    if eyes_closed_since is None:
                        eyes_closed_since = now
                    sleepy = now - eyes_closed_since >= args.closed_seconds
                else:
                    eyes_closed_since = None
            else:
                eyes_closed_since = None
            closed_elapsed = (
                max(0.0, now - eyes_closed_since)
                if eyes_closed_since is not None
                else 0.0
            )

            # Run the heavier YOLO model at a lower rate than face tracking.
            if now >= next_phone_check:
                results = model.predict(
                    frame,
                    classes=phone_ids,
                    conf=args.phone_confidence,
                    imgsz=640,
                    verbose=False,
                )
                found_boxes: list[tuple[int, int, int, int, float]] = []
                result = results[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                        confidence = float(box.conf[0].item())
                        found_boxes.append((x1, y1, x2, y2, confidence))
                cached_phone_boxes = found_boxes
                next_phone_check = time.monotonic() + args.phone_interval

            phone_detected = bool(cached_phone_boxes)
            if phone_detected:
                phone_alert_seconds += delta
            if sleepy:
                sleepy_alert_seconds += delta
            if face_missing:
                face_missing_alert_seconds += delta

            # Alert priority mirrors the demo: phone, face visibility, then drowsiness.
            if phone_detected:
                state = "PHONE"
                message, color = "PUT THE PHONE AWAY!", (0, 145, 255)
            elif face_missing:
                state = "FACE"
                message, color = "DON'T COVER YOUR FACE!", (0, 0, 220)
            elif sleepy:
                state = "SLEEPY"
                message, color = "WAKE UP & STUDY!", (0, 0, 220)
            else:
                state = "OK"
                message, color = "STUDY MODE: ON", (45, 150, 45)

            draw_banner(frame, message, color)
            if show_eye_debug and face_found:
                draw_eye_measurements(frame, faces[0])
            draw_eye_status_panel(
                frame,
                face_found,
                ratio,
                left_ratio,
                right_ratio,
                eye_threshold,
                closed_elapsed,
                args.closed_seconds,
            )
            draw_phone_boxes(frame, cached_phone_boxes)

            elapsed = time.monotonic() - session_start
            cv2.putText(frame, f"Session {format_duration(elapsed)}", (20, frame.shape[0] - 42),
                        FONT, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Q: quit   C: calibrate   D: eye points   [: lower   ]: raise threshold",
                        (20, frame.shape[0] - 16), FONT, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame)
            if state == "PHONE":
                alert_sound = phone_sound
            elif state == "SLEEPY":
                alert_sound = sleepy_sound
            else:
                alert_sound = beep
            if state != "OK":
                # Start a new alert immediately. If eye tracking briefly flickers
                # back to OK, let the spoken reminder finish instead of cutting it off.
                if state != active_audio_alert:
                    if audio_channel is not None:
                        audio_channel.stop()
                    active_audio_alert = state
                    audio_channel = alert_sound.play() if alert_sound is not None else None
                    last_audio_time = now
                    if state == "SLEEPY":
                        print("Playing eye-closed voice alert.")
                    elif state == "PHONE":
                        print("Playing phone-detected voice alert.")
                    elif state == "FACE":
                        print("Playing face-visibility beep alert.")
                elif (
                    alert_sound is not None
                    and (audio_channel is None or not audio_channel.get_busy())
                    and now - last_audio_time >= 2.0
                ):
                    audio_channel = alert_sound.play()
                    last_audio_time = now
            elif audio_channel is None or not audio_channel.get_busy():
                # An alert may clear before its voice clip finishes. Forget the
                # old alert only after playback has ended.
                audio_channel = None
                active_audio_alert = None

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                if ratio is not None and 0.14 <= ratio <= 0.40:
                    eye_threshold = min(0.35, max(0.08, ratio * 0.65))
                    print(f"Calibrated eye threshold to {eye_threshold:.3f} ({eye_threshold * 100:.0f}).")
                else:
                    print("Calibration skipped. Face the camera with both eyes naturally open, then press C.")
            if key == ord("d"):
                show_eye_debug = not show_eye_debug
                print(f"Eye landmark debug: {'on' if show_eye_debug else 'off'}.")
            if key == ord("]"):
                eye_threshold = min(0.40, eye_threshold + 0.01)
                print(f"Eye threshold: {eye_threshold:.3f}")
            if key == ord("["):
                eye_threshold = max(0.08, eye_threshold - 0.01)
                print(f"Eye threshold: {eye_threshold:.3f}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if pygame.mixer.get_init():
            pygame.mixer.quit()

    total = time.monotonic() - session_start
    print("\nSession summary")
    print(f"Total time:          {format_duration(total)}")
    print(f"Phone alert time:    {format_duration(phone_alert_seconds)}")
    print(f"Eyes-closed alerts:  {format_duration(sleepy_alert_seconds)}")
    print(f"Face-missing alerts: {format_duration(face_missing_alert_seconds)}")
    print("Camera frames were processed live and were not written to disk.")


if __name__ == "__main__":
    main()
