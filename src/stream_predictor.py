import time
from typing import Any, Dict


class StreamPredictor:
    """
    Iterates over new camera frames and runs detector inference.
    """

    def __init__(self, predictor, stream, idle_sleep: float = 0.005):
        self.predictor = predictor
        self.stream = stream
        self.idle_sleep = idle_sleep
        self._last_frame_id = 0

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, Any]:
        while True:
            frame, frame_id, frame_ts = self.stream.get_frame_with_meta(copy=False)

            if frame is None:
                if not self.stream.is_running:
                    raise StopIteration
                time.sleep(self.idle_sleep)
                continue

            if frame_id <= self._last_frame_id:
                if not self.stream.is_running:
                    raise StopIteration
                time.sleep(self.idle_sleep)
                continue

            self._last_frame_id = frame_id

            inference_start = time.perf_counter()
            detections = self.predictor.predict(frame)
            inference_ms = (time.perf_counter() - inference_start) * 1000.0

            return {
                "frame": frame,
                "frame_id": frame_id,
                "timestamp": frame_ts,
                "inference_ms": inference_ms,
                "detections": detections,
            }
