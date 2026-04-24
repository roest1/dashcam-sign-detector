"""OpenCV real-time frame loop for the dashcam-sign-detector pipeline.

Responsibilities:

- Open a ``cv2.VideoCapture`` from a webcam index, a file path, or an RTSP URL.
- For each frame, convert BGR → RGB, run the pipeline, draw bboxes and a
  rolling FPS overlay back onto the BGR frame.
- Optionally write the annotated frames to an MP4 via ``cv2.VideoWriter``.
- Optionally display in an interactive window. Headless-safe: if ``imshow``
  fails (no DISPLAY, container without X socket, CI), the loop logs a warning
  and continues without display.
- Return aggregate stats so callers (CLI, benchmarks, tests) can report them.

Pattern ported from ``FaceFinder/livefacetracker.py`` (previous tutorial
project kept as a reference — deleted at the end of Phase C).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from dashcam_sign_detector.pipeline.pipeline import PipelineResult

_DEFAULT_BBOX_COLOUR_BGR: tuple[int, int, int] = (0, 255, 0)  # lime
_DEFAULT_TEXT_COLOUR_BGR: tuple[int, int, int] = (0, 0, 0)
_DEFAULT_WINDOW_NAME = "dashcam-sign-detector"


class _SupportsRun(Protocol):
    """Structural type for any pipeline object exposing ``.run(rgb) -> list``."""

    def run(self, image: np.ndarray) -> list[PipelineResult]: ...


@dataclass(frozen=True)
class StreamStats:
    """Aggregate metrics returned by :func:`run_stream`."""

    frames_processed: int
    total_detections: int
    wall_time_seconds: float
    average_fps: float


def parse_source(source: int | str) -> int | str:
    """Normalise a user-supplied source string into a VideoCapture argument.

    - ``int`` values pass through as camera indices.
    - ``"webcam"`` is a convenience alias for camera index ``0``.
    - Numeric strings (``"0"``, ``"1"``) are parsed as camera indices.
    - Everything else is returned verbatim so it can be a file path, an
      ``rtsp://``/``http://`` URL, or a device node like ``/dev/video0``
      (on Linux ``cv2.VideoCapture`` accepts that as a string).
    """
    if isinstance(source, int):
        return source
    if not isinstance(source, str):
        raise TypeError(f"Expected int or str source, got {type(source).__name__}")
    if source.lower() == "webcam":
        return 0
    try:
        return int(source)
    except ValueError:
        return source


def draw_detections(
    frame_bgr: np.ndarray,
    results: list[PipelineResult],
    *,
    colour: tuple[int, int, int] = _DEFAULT_BBOX_COLOUR_BGR,
) -> None:
    """Draw bboxes + ``class (conf)`` labels onto a BGR frame, in place."""
    for r in results:
        x1, y1, x2, y2 = r.bbox
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), colour, 2)
        label = f"{r.classifier_class} {r.classifier_confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        top = max(0, y1 - text_h - baseline - 4)
        cv2.rectangle(
            frame_bgr,
            (x1, top),
            (x1 + text_w + 6, top + text_h + baseline + 4),
            colour,
            thickness=-1,
        )
        cv2.putText(
            frame_bgr,
            label,
            (x1 + 3, top + text_h + baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _DEFAULT_TEXT_COLOUR_BGR,
            1,
            cv2.LINE_AA,
        )


def draw_fps(frame_bgr: np.ndarray, fps: float) -> None:
    """Overlay a rolling FPS readout in the top-left corner."""
    text = f"{fps:5.1f} FPS"
    origin = (10, 24)
    cv2.putText(
        frame_bgr, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 0, 0), 3, cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 1, cv2.LINE_AA,
    )


def annotate_frame(
    frame_bgr: np.ndarray,
    results: list[PipelineResult],
    *,
    fps: float | None = None,
) -> None:
    """Convenience: draw both detections and (optionally) FPS in one call."""
    draw_detections(frame_bgr, results)
    if fps is not None:
        draw_fps(frame_bgr, fps)


def run_stream(
    pipeline: _SupportsRun,
    source: int | str,
    *,
    output: Path | str | None = None,
    display: bool = True,
    max_frames: int | None = None,
    window_name: str = _DEFAULT_WINDOW_NAME,
    quit_key: str = "q",
) -> StreamStats:
    """Iterate frames from ``source``, run the pipeline, and emit annotated output.

    The pipeline is duck-typed: anything with a ``run(rgb_array) -> list[PipelineResult]``
    method works, which is exactly what :class:`DetectionClassificationPipeline`
    provides. Tests inject a fake here.

    Returns a :class:`StreamStats` record. Raises ``RuntimeError`` if the
    capture or writer cannot be opened.
    """
    capture_source = parse_source(source)
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source {source!r}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    writer: cv2.VideoWriter | None = None
    if output is not None:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, float(fps_in), (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open VideoWriter at {out_path}")

    frames_processed = 0
    total_detections = 0
    fps_window: deque[float] = deque(maxlen=30)
    start = time.perf_counter()
    last_tick = start
    quit_ord = ord(quit_key)
    display_enabled = display

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = pipeline.run(rgb)
            total_detections += len(results)

            now = time.perf_counter()
            instantaneous_fps = 1.0 / max(now - last_tick, 1e-6)
            last_tick = now
            fps_window.append(instantaneous_fps)
            rolling_fps = sum(fps_window) / len(fps_window)

            annotate_frame(frame_bgr, results, fps=rolling_fps)

            if writer is not None:
                writer.write(frame_bgr)

            if display_enabled:
                try:
                    cv2.imshow(window_name, frame_bgr)
                    if (cv2.waitKey(1) & 0xFF) == quit_ord:
                        break
                except cv2.error as exc:
                    print(
                        f"[dashcam-sign-detector] imshow failed ({exc}); "
                        "continuing headless."
                    )
                    display_enabled = False

            frames_processed += 1
            if max_frames is not None and frames_processed >= max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    wall = time.perf_counter() - start
    return StreamStats(
        frames_processed=frames_processed,
        total_detections=total_detections,
        wall_time_seconds=wall,
        average_fps=frames_processed / max(wall, 1e-6),
    )
