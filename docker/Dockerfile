# Turret Vision v2 runtime image. Built by CI on GitHub's arm64 runners and pulled by
# the Jetson, so the Jetson never compiles anything and never touches its own Python.
#
# Why plain Ubuntu 22.04 (not l4t-jetpack) for now: nothing through M3 needs CUDA.
# Vision runs on the CPU (OpenCV), video is the camera's own JPEG. The TensorRT layer
# arrives with the learned detector in M4, when the base switches to l4t-jetpack:r36.4.0
# and the NVMe drive has room for it. Ubuntu 22.04 matches JetPack 6 (Python 3.10).

# ---- GUI build -------------------------------------------------------------------
FROM node:20-bookworm-slim AS ui
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY ui/ ./
RUN npm run build

# ---- runtime ---------------------------------------------------------------------
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip v4l-utils ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/turret
COPY requirements.lock ./
RUN pip3 install -r requirements.lock && pip3 install "platformio==6.1.18"
COPY pyproject.toml README.md ./
COPY turret/ turret/
RUN pip3 install --no-deps . \
 && python3 -c "import cv2, numpy; assert numpy.__version__.startswith('1.'); print('opencv', cv2.__version__, 'numpy', numpy.__version__)"
COPY proto/ proto/
COPY tools/ tools/
COPY firmware/ firmware/
COPY tests/ tests/
COPY --from=ui /ui/dist /opt/turret/ui/dist

# /data is the bind-mounted data/ folder: sessions, local.yaml, PlatformIO toolchains.
ENV TURRET_UI_DIR=/opt/turret/ui/dist \
    TURRET_DATA=/data \
    PLATFORMIO_CORE_DIR=/data/.platformio
EXPOSE 8080
CMD ["python3", "-m", "turret", "serve"]
