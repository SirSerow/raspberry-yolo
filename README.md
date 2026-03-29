# Raspberry Pi Camera Detection (Ultralytics)

This app runs real-time object detection from a Raspberry Pi camera source.

- Backend: Ultralytics (`yolo11n` by default)
- Camera input: Picamera2 (CSI) with OpenCV fallback (USB/video/stream)
- Output (MVP): console logging + CSV detections

## Project Status

This is now runnable end-to-end via `main.py` with config-driven behavior.

Implemented in MVP:
- Config loading and validation from `config.yml`
- Camera backend selection: `auto | picamera2 | opencv`
- Real-time inference loop and per-frame timing logs
- CSV detection output when `output.save_csv: true`
- Graceful shutdown on `Ctrl+C`

Not implemented in MVP:
- On-screen display rendering (`output.display`)
- Annotated video saving (`output.save_video`)

## Configuration

Main file: `config.yml`

Important keys:
- `camera.backend`: `auto`, `picamera2`, or `opencv`
- `camera.source`: OpenCV source (device index/path/URL), ignored for Picamera2
- `model.model_name`: model name, `.pt` is auto-appended if missing
- `model.weights_path`: file or directory override for local weights
- `output.save_csv`: enable CSV generation in `output.output_dir`

Model resolution precedence:
1. If `weights_path` is an existing file: use it
2. If `weights_path` is a directory and `<model_name>.pt` exists there: use it
3. Else use `model_name` (Ultralytics may auto-download)

## Run Natively (Raspberry Pi)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py --config config.yml
```

Optional test cap:

```bash
python3 main.py --config config.yml --max-frames 300
```

## Docker (Raspberry Pi OS)

Build:

```bash
docker build -t raspi-cam-yolo .
```

If you change the Dockerfile or requirements, rebuild the image before running again.
The CSI/Picamera2 image build also pulls packages from the Raspberry Pi package archive.

Run with USB/OpenCV camera (`/dev/video0`):

```bash
docker run --rm -it \
  --device /dev/video0:/dev/video0 \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/logs:/app/logs" \
  raspi-cam-yolo python3 main.py --config config.yml
```

Run with CSI/Picamera2. Set `camera.backend: picamera2` in `config.yml` and use broader device/libcamera access on Pi:

```bash
docker run --rm -it --privileged \
  -v /run/udev:/run/udev:ro \
  -v /dev:/dev \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/logs:/app/logs" \
  raspi-cam-yolo python3 main.py --config config.yml
```

CSI camera smoke test in the container:

```bash
docker run --rm -it --privileged \
  -v /run/udev:/run/udev:ro \
  -v /dev:/dev \
  raspi-cam-yolo python3 tests/camera_smoke_test.py --backend picamera2
```

If CSI is unavailable in container, set `camera.backend: opencv` and use a USB device source instead.

## CSV Output

When enabled, files are written as:

`output/detections_YYYYMMDD_HHMMSS.csv`

Columns:
- `timestamp`
- `frame_id`
- `inference_ms`
- `class_id`
- `class_name`
- `score`
- `bbox_xyxy`
