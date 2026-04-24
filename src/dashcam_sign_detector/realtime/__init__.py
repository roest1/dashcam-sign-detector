"""Real-time OpenCV loop for dashcam frame processing."""

from dashcam_sign_detector.realtime.stream import (
    StreamStats,
    annotate_frame,
    draw_detections,
    draw_fps,
    parse_source,
    run_stream,
)

__all__ = [
    "StreamStats",
    "annotate_frame",
    "draw_detections",
    "draw_fps",
    "parse_source",
    "run_stream",
]
