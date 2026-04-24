"""CLI entry point for the realtime dashcam pipeline.

Examples::

    # Webcam (default device 0), interactive window, press 'q' to quit
    python -m dashcam_sign_detector.realtime

    # Process a video file headlessly and write an annotated MP4
    python -m dashcam_sign_detector.realtime \\
        --input sample.mp4 --output annotated.mp4 --no-display

    # RTSP stream on CPU with the fast SSDLite backbone
    python -m dashcam_sign_detector.realtime \\
        --input rtsp://camera.local/stream \\
        --detector ssdlite320_mobilenet_v3_large \\
        --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dashcam_sign_detector.detector.detect import list_available_models
from dashcam_sign_detector.pipeline import build_default_pipeline
from dashcam_sign_detector.realtime.stream import run_stream


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dashcam_sign_detector.realtime",
        description="Run the dashcam-sign-detector pipeline on a live or recorded video stream.",
    )
    parser.add_argument(
        "--input",
        default="webcam",
        help=(
            "Source for cv2.VideoCapture. Accepts 'webcam' (alias for 0), "
            "a camera index ('0'), a file path, or an rtsp://... URL. "
            "Default: webcam."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write an annotated MP4 via cv2.VideoWriter.",
    )
    parser.add_argument(
        "--detector",
        default=None,
        choices=list_available_models(),
        help="Torchvision detector backbone. Default: fasterrcnn_resnet50_fpn_v2.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Detector confidence floor (default 0.5).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override (e.g. 'cpu', 'cuda'). Default: auto.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable interactive cv2.imshow window (use for headless/container runs).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after processing this many frames. Useful for benchmarks.",
    )
    parser.add_argument(
        "--classifier-path",
        type=Path,
        default=None,
        help="Override the FastAI classifier pkl (default: ClassifierConfig().model_path).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    pipeline = build_default_pipeline(
        detector_model=args.detector,
        score_threshold=args.score_threshold,
        classifier_path=args.classifier_path,
        device=args.device,
    )

    stats = run_stream(
        pipeline,
        source=args.input,
        output=args.output,
        display=not args.no_display,
        max_frames=args.max_frames,
    )

    print(f"Frames processed : {stats.frames_processed}")
    print(f"Total detections : {stats.total_detections}")
    print(f"Wall time        : {stats.wall_time_seconds:.2f} s")
    print(f"Average FPS      : {stats.average_fps:.2f}")
    if args.output is not None:
        print(f"Annotated output : {args.output}")


if __name__ == "__main__":
    main()
