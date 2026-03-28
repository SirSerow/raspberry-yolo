from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    from loguru import logger
except ImportError:  # pragma: no cover - dependency may be absent before install
    logger = None


class ConfigError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-time Ultralytics detection on Raspberry Pi camera streams."
    )
    parser.add_argument(
        "--config",
        default="config.yml",
        help="Path to YAML config file (default: config.yml)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional safety cap for number of frames to process.",
    )
    return parser.parse_args()


def _to_bool(value: Any, key: str, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ConfigError(f"Invalid boolean for '{key}': {value!r}")


def _required_section(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    section = config.get(name)
    if section is None:
        raise ConfigError(f"Missing config section: '{name}'")
    if not isinstance(section, dict):
        raise ConfigError(f"Config section '{name}' must be a mapping")
    return section


def _number(section: Dict[str, Any], key: str, cast_type, min_value: Optional[float] = None):
    if key not in section:
        raise ConfigError(f"Missing required key: '{key}'")
    try:
        value = cast_type(section[key])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid value for '{key}': {section[key]!r}") from exc

    if min_value is not None and value < min_value:
        raise ConfigError(f"Invalid value for '{key}': must be >= {min_value}")
    return value


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed parsing YAML config: {exc}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError("Root config must be a mapping")
    return loaded


def normalize_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    camera = _required_section(raw, "camera")
    model = _required_section(raw, "model")
    output = _required_section(raw, "output")
    logging_cfg = _required_section(raw, "logging")

    backend = str(camera.get("backend", "auto")).strip().lower()
    if backend not in {"auto", "picamera2", "opencv"}:
        raise ConfigError("camera.backend must be one of: auto, picamera2, opencv")

    source = camera.get("source", 0)
    if isinstance(source, str):
        stripped = source.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            source = int(stripped)
        else:
            source = stripped

    width = _number(camera, "width", int, min_value=1)
    height = _number(camera, "height", int, min_value=1)
    fps = _number(camera, "fps", int, min_value=1)

    try:
        conf = float(model.get("conf_threshold", 0.5))
        iou = float(model.get("nms_threshold", 0.4))
    except (TypeError, ValueError) as exc:
        raise ConfigError("model.conf_threshold and model.nms_threshold must be numeric") from exc

    if not (0.0 <= conf <= 1.0):
        raise ConfigError("model.conf_threshold must be between 0 and 1")
    if not (0.0 <= iou <= 1.0):
        raise ConfigError("model.nms_threshold must be between 0 and 1")

    model_name = str(model.get("model_name", "yolo11n")).strip()
    if not model_name:
        raise ConfigError("model.model_name cannot be empty")
    weights_path = model.get("weights_path")
    device = str(model.get("device", "cpu")).strip()
    measure_fps = _to_bool(
        model.get("measure_inference_fps", True),
        "model.measure_inference_fps",
        default=True,
    )

    display = _to_bool(output.get("display", False), "output.display", default=False)
    save_video = _to_bool(output.get("save_video", False), "output.save_video", default=False)
    save_csv = _to_bool(output.get("save_csv", False), "output.save_csv", default=False)
    output_dir_raw = output.get("output_dir", "output")
    output_dir = None if output_dir_raw is None else Path(str(output_dir_raw))

    log_dir_raw = logging_cfg.get("log_dir", "logs")
    log_dir = Path(str(log_dir_raw))
    log_level = str(logging_cfg.get("log_level", "INFO")).upper()
    log_inference_times = _to_bool(
        logging_cfg.get("log_inference_times", True),
        "logging.log_inference_times",
        default=True,
    )
    log_rotation = _to_bool(
        logging_cfg.get("log_rotation", True),
        "logging.log_rotation",
        default=True,
    )

    return {
        "camera": {
            "backend": backend,
            "source": source,
            "width": width,
            "height": height,
            "fps": fps,
        },
        "model": {
            "model_name": model_name,
            "weights_path": weights_path,
            "conf_threshold": conf,
            "nms_threshold": iou,
            "device": device,
            "measure_inference_fps": measure_fps,
        },
        "output": {
            "display": display,
            "save_video": save_video,
            "save_csv": save_csv,
            "output_dir": output_dir,
        },
        "logging": {
            "log_dir": log_dir,
            "log_level": log_level,
            "log_inference_times": log_inference_times,
            "log_rotation": log_rotation,
        },
    }


def configure_logging(logging_cfg: Dict[str, Any]) -> None:
    if logger is None:
        raise RuntimeError("loguru is not installed. Run: pip install -r requirements.txt")

    log_dir = logging_cfg["log_dir"]
    log_level = logging_cfg["log_level"]
    rotation_enabled = logging_cfg["log_rotation"]

    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level=log_level, enqueue=True)

    app_log_path = log_dir / "app.log"
    if rotation_enabled:
        logger.add(
            app_log_path,
            level=log_level,
            rotation="10 MB",
            retention="7 days",
            enqueue=True,
        )
    else:
        logger.add(app_log_path, level=log_level, enqueue=True)


def resolve_model_reference(model_cfg: Dict[str, Any]) -> str:
    model_name = model_cfg["model_name"]
    weights_path = model_cfg.get("weights_path")
    model_name_with_suffix = model_name if Path(model_name).suffix else f"{model_name}.pt"

    if weights_path:
        weights_candidate = Path(str(weights_path))
        if weights_candidate.is_file():
            return str(weights_candidate)

        if weights_candidate.is_dir():
            nested_candidate = weights_candidate / model_name_with_suffix
            if nested_candidate.is_file():
                return str(nested_candidate)

    return model_name_with_suffix


def _format_bbox(bbox: Any) -> str:
    if not bbox:
        return ""
    return ",".join(f"{float(value):.2f}" for value in bbox)


def _open_csv_writer(output_dir: Path) -> tuple[csv.DictWriter, Any, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    handle = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "timestamp",
            "frame_id",
            "inference_ms",
            "class_id",
            "class_name",
            "score",
            "bbox_xyxy",
        ],
    )
    writer.writeheader()
    return writer, handle, csv_path


def run(config: Dict[str, Any], max_frames: Optional[int] = None) -> int:
    try:
        from src.stream_predictor import StreamPredictor
        from src.utils.camera_reader import CameraReader
        from src.utils.model_manager import UltralyticsDetector
    except ImportError as exc:
        raise RuntimeError(
            f"Missing runtime dependency: {exc}. Run: pip install -r requirements.txt"
        ) from exc

    camera_cfg = config["camera"]
    model_cfg = config["model"]
    output_cfg = config["output"]
    logging_cfg = config["logging"]

    configure_logging(logging_cfg)

    if output_cfg["display"]:
        logger.warning("output.display is not implemented in this MVP and will be ignored.")
    if output_cfg["save_video"]:
        logger.warning("output.save_video is not implemented in this MVP and will be ignored.")

    csv_writer = None
    csv_handle = None

    if output_cfg["save_csv"]:
        output_dir = output_cfg["output_dir"]
        if output_dir is None:
            raise ConfigError("output.output_dir cannot be null when output.save_csv=true")
        csv_writer, csv_handle, csv_path = _open_csv_writer(output_dir)
        logger.info(f"CSV output enabled: {csv_path}")

    camera = CameraReader(
        source=camera_cfg["source"],
        size=(camera_cfg["width"], camera_cfg["height"]),
        fps=camera_cfg["fps"],
        backend=camera_cfg["backend"],
    )

    model_reference = resolve_model_reference(model_cfg)
    logger.info(f"Loading model: {model_reference}")

    detector = UltralyticsDetector(
        model_path=model_reference,
        conf=model_cfg["conf_threshold"],
        iou=model_cfg["nms_threshold"],
        device=model_cfg["device"],
    )

    processed_frames = 0
    total_detections = 0
    started_at = time.perf_counter()
    last_report_time = started_at

    try:
        camera.start()
        first_frame = camera.wait_for_frame(timeout=5.0)
        if first_frame is None:
            raise RuntimeError("Camera started but no frames were received within timeout.")

        logger.info(
            f"Camera stream active (backend={camera.active_backend}, source={camera_cfg['source']!r})"
        )
        stream = StreamPredictor(detector, camera)

        for result in stream:
            frame_id = result["frame_id"]
            timestamp = result["timestamp"]
            inference_ms = result["inference_ms"]
            detections = result["detections"]

            processed_frames += 1
            total_detections += len(detections)

            if csv_writer and detections:
                iso_time = datetime.fromtimestamp(timestamp).isoformat()
                for det in detections:
                    csv_writer.writerow(
                        {
                            "timestamp": iso_time,
                            "frame_id": frame_id,
                            "inference_ms": f"{inference_ms:.3f}",
                            "class_id": det["class_id"],
                            "class_name": det["class_name"],
                            "score": f"{det['score']:.5f}",
                            "bbox_xyxy": _format_bbox(det["bbox"]),
                        }
                    )
                csv_handle.flush()

            if logging_cfg["log_inference_times"]:
                logger.debug(
                    f"frame={frame_id} detections={len(detections)} inference_ms={inference_ms:.2f}"
                )

            now = time.perf_counter()
            if now - last_report_time >= 1.0:
                elapsed = now - started_at
                if model_cfg["measure_inference_fps"] and elapsed > 0:
                    fps = processed_frames / elapsed
                    logger.info(
                        f"processed_frames={processed_frames} total_detections={total_detections} fps={fps:.2f}"
                    )
                else:
                    logger.info(
                        f"processed_frames={processed_frames} total_detections={total_detections}"
                    )
                last_report_time = now

            if max_frames is not None and processed_frames >= max_frames:
                logger.info(f"Reached frame limit (--max-frames={max_frames}).")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down.")
    finally:
        camera.stop()
        if csv_handle:
            csv_handle.close()

    logger.info(
        f"Finished. processed_frames={processed_frames}, total_detections={total_detections}"
    )
    return 0


def main() -> int:
    args = parse_args()

    try:
        raw_config = load_config(Path(args.config))
        config = normalize_config(raw_config)
        return run(config, max_frames=args.max_frames)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Runtime error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
