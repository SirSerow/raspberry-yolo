from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - dependency availability differs by env
    YOLO = None


class UltralyticsDetector:
    """
    Simple Ultralytics detector manager for use with CameraReader frames.

    - Loads model once
    - Auto-downloads official weights if not present
    - Accepts OpenCV BGR numpy frames
    - Returns simple detection dicts
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
    ) -> None:
        if YOLO is None:
            raise RuntimeError(
                "ultralytics is not installed. Add it to requirements and install dependencies."
            )

        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.device = device

        self.model = YOLO(self.model_path)

    def predict(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        results = self.model(
            frame_bgr,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []

        for result in results:
            names = result.names
            if result.boxes is None:
                continue

            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())

                if isinstance(names, dict):
                    class_name = names.get(cls_id, str(cls_id))
                elif isinstance(names, list) and 0 <= cls_id < len(names):
                    class_name = names[cls_id]
                else:
                    class_name = str(cls_id)

                detections.append(
                    {
                        "bbox": [float(v) for v in xyxy],
                        "score": conf,
                        "class_id": cls_id,
                        "class_name": class_name,
                    }
                )

        return detections
