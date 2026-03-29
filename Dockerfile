FROM debian:bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-picamera2 \
    python3-libcamera \
    libcamera-apps \
    build-essential \
    gcc \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3 -m pip install --break-system-packages --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "main.py", "--config", "config.yml"]
