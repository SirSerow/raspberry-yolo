from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import load_config, normalize_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test Raspberry Pi/USB camera setup without loading the detector."
    )
    parser.add_argument(
        "--config",
        default="config.yml",
        help="Path to app config (default: config.yml)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "picamera2", "opencv"],
        default=None,
        help="Override camera backend from config",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Override camera source from config for OpenCV backend",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=8.0,
        help="Seconds to wait for first frame (default: 8)",
    )
    parser.add_argument(
        "--direct-opencv-probe",
        action="store_true",
        help="Probe OpenCV VideoCapture directly before using CameraReader",
    )
    return parser.parse_args()


def normalize_source(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            return int(stripped)
        return stripped
    return value


def direct_opencv_probe(source: Any, width: int, height: int, fps: int) -> bool:
    print(f"[probe] opening OpenCV source={source!r}")
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            print("[probe] FAIL: VideoCapture did not open.")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        cap.set(cv2.CAP_PROP_FPS, float(fps))

        ok, frame = cap.read()
        if not ok or frame is None:
            print("[probe] FAIL: device opened but returned no frame on first read.")
            return False

        print(
            f"[probe] OK: first frame shape={frame.shape}, dtype={frame.dtype}, size={frame.size}"
        )
        return True
    finally:
        cap.release()


def main() -> int:
    args = parse_args()

    try:
        from src.utils.camera_reader import CameraReader
    except ImportError as exc:
        print(
            f"[camera-test] FAIL: missing runtime dependency: {exc}. "
            "Run: pip install -r requirements.txt"
        )
        return 2

    raw = load_config(Path(args.config))
    config = normalize_config(raw)
    camera_cfg = config["camera"]

    backend = args.backend or camera_cfg["backend"]
    source = normalize_source(args.source) if args.source is not None else camera_cfg["source"]
    width = camera_cfg["width"]
    height = camera_cfg["height"]
    fps = camera_cfg["fps"]

    print("[camera-test] requested settings")
    print(f"[camera-test] backend={backend}")
    print(f"[camera-test] source={source!r}")
    print(f"[camera-test] size={width}x{height}")
    print(f"[camera-test] fps={fps}")

    if args.direct_opencv_probe or backend == "opencv":
        probe_ok = direct_opencv_probe(source, width, height, fps)
        if not probe_ok and backend == "opencv":
            print("[camera-test] OpenCV probe failed before CameraReader startup.")

    reader = CameraReader(
        source=source,
        size=(width, height),
        fps=fps,
        backend=backend,
        frame_timeout=args.timeout,
    )

    try:
        print("[camera-test] starting CameraReader")
        reader.start()
        print(f"[camera-test] active_backend={reader.active_backend}")

        deadline = time.time() + args.timeout
        last_frame_id = 0

        while time.time() < deadline:
            frame, frame_id, timestamp = reader.get_frame_with_meta(copy=False)
            if frame is not None:
                age_ms = (time.time() - timestamp) * 1000.0 if timestamp else -1.0
                print(
                    "[camera-test] OK: "
                    f"frame_id={frame_id} shape={frame.shape} dtype={frame.dtype} "
                    f"age_ms={age_ms:.1f}"
                )
                return 0

            if frame_id != last_frame_id:
                print(f"[camera-test] observed frame counter change: {frame_id}")
                last_frame_id = frame_id

            time.sleep(0.05)

        print("[camera-test] FAIL: no frame received before timeout.")
        print("[camera-test] checks:")
        print("[camera-test] - verify the correct backend is selected")
        print("[camera-test] - for Docker/OpenCV, confirm /dev/video0 is passed through")
        print("[camera-test] - confirm no other process is holding the camera device")
        print("[camera-test] - on Raspberry Pi CSI, prefer backend=picamera2")
        return 1
    except Exception as exc:
        print(f"[camera-test] FAIL: {exc}")
        return 2
    finally:
        reader.stop()


if __name__ == "__main__":
    raise SystemExit(main())
