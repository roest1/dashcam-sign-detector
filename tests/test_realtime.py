"""Tests for the realtime OpenCV stream loop.

These tests avoid any real model load: they drive ``run_stream`` against a
tempfile MP4 built with ``cv2.VideoWriter`` and a fake pipeline that returns
deterministic detections. Display is always disabled so the tests are
headless-safe in CI.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from dashcam_sign_detector.pipeline.pipeline import PipelineResult
from dashcam_sign_detector.realtime.stream import (
    StreamStats,
    draw_detections,
    draw_fps,
    parse_source,
    run_stream,
)

# ---------- parse_source ------------------------------------------------------


def test_parse_source_webcam_alias():
    assert parse_source("webcam") == 0
    assert parse_source("WEBCAM") == 0


def test_parse_source_integer_passthrough():
    assert parse_source(0) == 0
    assert parse_source(2) == 2


def test_parse_source_numeric_string_becomes_index():
    assert parse_source("0") == 0
    assert parse_source("3") == 3


def test_parse_source_path_returns_string():
    assert parse_source("/tmp/video.mp4") == "/tmp/video.mp4"
    assert parse_source("rtsp://camera.local/stream") == "rtsp://camera.local/stream"


def test_parse_source_rejects_bad_type():
    with pytest.raises(TypeError):
        parse_source(3.14)  # type: ignore[arg-type]


# ---------- draw_detections ---------------------------------------------------


def _blank_bgr(h: int = 120, w: int = 160) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_draw_detections_modifies_pixels_inside_bbox():
    frame = _blank_bgr()
    result = PipelineResult(
        bbox=(20, 30, 80, 90),
        detector_class="stop sign",
        detector_score=0.9,
        classifier_class="14",
        classifier_confidence=0.87,
    )

    draw_detections(frame, [result])

    # Bounding rectangle edges should be non-zero somewhere along them.
    top_edge = frame[30, 20:80]
    assert top_edge.any()
    # Interior (excluding the drawn rectangle line + label rectangle) should
    # remain black; check a point well inside the bbox but below any labels.
    assert not frame[60, 50].any()


def test_draw_detections_noop_on_empty_list():
    frame = _blank_bgr()
    draw_detections(frame, [])
    assert not frame.any()


def test_draw_fps_overlays_text():
    frame = _blank_bgr()
    draw_fps(frame, 42.0)
    # Some pixels in the top band must become non-zero after text render.
    assert frame[:40, :120].any()


# ---------- run_stream integration via tempfile MP4 --------------------------


class _FakePipeline:
    """Deterministic pipeline: emits one full-frame detection per frame."""

    def __init__(self) -> None:
        self.frames_seen = 0
        self.last_shape: tuple[int, int, int] | None = None

    def run(self, rgb: np.ndarray):
        self.frames_seen += 1
        self.last_shape = rgb.shape
        h, w, _ = rgb.shape
        return [
            PipelineResult(
                bbox=(10, 10, w - 10, h - 10),
                detector_class="stop sign",
                detector_score=0.95,
                classifier_class="14",
                classifier_confidence=0.88,
            )
        ]


def _write_gradient_video(path: Path, *, frames: int = 8, h: int = 120, w: int = 160) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 15.0, (w, h))
    assert writer.isOpened(), "opencv could not open mp4v writer in test"
    try:
        for i in range(frames):
            frame = np.full((h, w, 3), fill_value=(i * 20) % 256, dtype=np.uint8)
            # Stamp a channel ramp so successive frames differ after encode.
            frame[:, :, 0] = (np.arange(w) + i * 3) % 256
            writer.write(frame)
    finally:
        writer.release()


def test_run_stream_processes_all_frames_of_temp_video(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    _write_gradient_video(video, frames=6)

    pipeline = _FakePipeline()
    stats = run_stream(pipeline, source=str(video), display=False)

    assert isinstance(stats, StreamStats)
    assert stats.frames_processed == 6
    assert stats.total_detections == 6  # one fake detection per frame
    assert pipeline.frames_seen == 6
    assert stats.average_fps > 0
    assert pipeline.last_shape is not None and pipeline.last_shape[2] == 3


def test_run_stream_writes_annotated_mp4(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    output = tmp_path / "annotated.mp4"
    _write_gradient_video(video, frames=5)

    pipeline = _FakePipeline()
    stats = run_stream(
        pipeline, source=str(video), output=output, display=False
    )

    assert stats.frames_processed == 5
    assert output.exists()
    assert output.stat().st_size > 0

    # Re-open the written file to confirm it is a valid, non-empty video.
    reopened = cv2.VideoCapture(str(output))
    try:
        assert reopened.isOpened()
        read_frames = 0
        while True:
            ok, _ = reopened.read()
            if not ok:
                break
            read_frames += 1
        assert read_frames >= 1
    finally:
        reopened.release()


def test_run_stream_respects_max_frames(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    _write_gradient_video(video, frames=10)

    pipeline = _FakePipeline()
    stats = run_stream(
        pipeline, source=str(video), display=False, max_frames=3
    )

    assert stats.frames_processed == 3
    assert pipeline.frames_seen == 3


def test_run_stream_raises_on_bad_source(tmp_path: Path):
    bogus = tmp_path / "does-not-exist.mp4"
    pipeline = _FakePipeline()

    with pytest.raises(RuntimeError, match="Could not open"):
        run_stream(pipeline, source=str(bogus), display=False)
