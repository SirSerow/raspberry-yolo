from __future__ import annotations

import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger

try:
    from picamera2 import Picamera2
except ImportError:  # pragma: no cover - depends on target hardware/software stack
    Picamera2 = None


class CameraReader:
    """
    Background camera reader for Raspberry Pi and USB cameras.

    Designed for inference pipelines:
    - continuously grabs frames in a background thread
    - keeps only the latest frame
    - returns frames as BGR numpy arrays
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        size: Tuple[int, int] = (640, 480),
        fps: int = 30,
        backend: str = "auto",
        warmup_seconds: float = 1.0,
        frame_timeout: float = 2.0,
    ) -> None:
        self.source = source
        self.size = size
        self.fps = fps
        self.backend = backend
        self.warmup_seconds = warmup_seconds
        self.frame_timeout = frame_timeout

        self._picam2: Optional[Picamera2] = None
        self._capture: Optional[cv2.VideoCapture] = None
        self._capture_is_file = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._active_backend: Optional[str] = None

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._latest_frame_id: int = 0
        self._running: bool = False

    def start(self) -> None:
        if self._running:
            return

        self._stop_event.clear()
        errors = []

        for candidate in self._backend_candidates():
            try:
                if candidate == "picamera2":
                    self._start_picamera2()
                elif candidate == "opencv":
                    self._start_opencv()
                else:
                    raise RuntimeError(f"Unknown backend candidate: {candidate}")
                self._active_backend = candidate
                logger.info(f"[CameraReader] started using backend={candidate}")
                break
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
                self._teardown_backend()

        if self._active_backend is None:
            joined = " | ".join(errors) if errors else "no camera backend available"
            raise RuntimeError(f"Unable to start camera. {joined}")

        time.sleep(self.warmup_seconds)

        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        self._running = True

    def stop(self) -> None:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        self._teardown_backend()

        with self._lock:
            self._latest_frame = None
            self._latest_timestamp = 0.0

        self._active_backend = None
        self._running = False

    def _backend_candidates(self) -> Tuple[str, ...]:
        requested = str(self.backend).strip().lower()
        if requested == "auto":
            return ("picamera2", "opencv")
        if requested in {"picamera2", "opencv"}:
            return (requested,)
        raise ValueError(
            f"Unsupported camera backend '{self.backend}'. Use auto|picamera2|opencv."
        )

    def _start_picamera2(self) -> None:
        if Picamera2 is None:
            raise RuntimeError("Picamera2 is not installed in this environment.")

        self._picam2 = Picamera2()
        config = self._picam2.create_video_configuration(
            main={"size": self.size, "format": "RGB888"},
            buffer_count=4,
            controls={
                "FrameDurationLimits": (
                    int(1_000_000 / self.fps),
                    int(1_000_000 / self.fps),
                )
            },
        )
        self._picam2.configure(config)
        self._picam2.start()

    def _start_opencv(self) -> None:
        source = self.source
        if isinstance(source, str):
            stripped = source.strip()
            if stripped.isdigit() or (
                stripped.startswith("-") and stripped[1:].isdigit()
            ):
                source = int(stripped)

        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            raise RuntimeError(f"OpenCV failed to open source={source!r}")

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.size[0]))
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.size[1]))
        self._capture.set(cv2.CAP_PROP_FPS, float(self.fps))
        self._capture_is_file = isinstance(source, str)

    def _teardown_backend(self) -> None:
        if self._picam2 is not None:
            try:
                self._picam2.stop()
            finally:
                self._picam2.close()
                self._picam2 = None

        if self._capture is not None:
            self._capture.release()
            self._capture = None
            self._capture_is_file = False

    def _reader_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    frame = self._read_frame()
                    if frame is None:
                        continue

                    now = time.time()
                    with self._lock:
                        self._latest_frame = frame
                        self._latest_timestamp = now
                        self._latest_frame_id += 1

                except Exception as exc:
                    logger.error(f"[CameraReader] frame read error: {exc}")
                    time.sleep(0.05)
        finally:
            self._running = False

    def _read_frame(self) -> Optional[np.ndarray]:
        if self._active_backend == "picamera2":
            if self._picam2 is None:
                return None
            rgb = self._picam2.capture_array()
            if rgb is None:
                return None
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if self._active_backend == "opencv":
            if self._capture is None:
                return None

            ok, frame = self._capture.read()
            if not ok or frame is None:
                if self._capture_is_file:
                    logger.info("[CameraReader] reached end of video source")
                    self._stop_event.set()
                else:
                    time.sleep(0.01)
                return None
            return frame

        return None

    def get_frame(self, copy: bool = False) -> Optional[np.ndarray]:
        """
        Return the latest frame or None if no frame is available yet.

        Args:
            copy: If True, returns a copy of the frame.
                  If False, returns the internal latest frame reference.
                  Use False for best performance if the consumer does not modify the image.
        """
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy() if copy else self._latest_frame

    def get_frame_with_meta(
        self, copy: bool = False
    ) -> Tuple[Optional[np.ndarray], int, float]:
        """
        Return (frame, frame_id, timestamp).
        """
        with self._lock:
            if self._latest_frame is None:
                return None, self._latest_frame_id, self._latest_timestamp

            frame = self._latest_frame.copy() if copy else self._latest_frame
            return frame, self._latest_frame_id, self._latest_timestamp

    def wait_for_frame(
        self, timeout: Optional[float] = None, copy: bool = False
    ) -> Optional[np.ndarray]:
        """
        Wait until at least one frame is available.
        """
        deadline = time.time() + (
            timeout if timeout is not None else self.frame_timeout
        )

        while time.time() < deadline:
            frame = self.get_frame(copy=copy)
            if frame is not None:
                return frame
            time.sleep(0.01)

        return None

    def read(self, copy: bool = False) -> Optional[np.ndarray]:
        """
        Compatibility alias for stream-like callers.
        """
        return self.get_frame(copy=copy)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def active_backend(self) -> Optional[str]:
        return self._active_backend
