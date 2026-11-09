# The devel image includes nvcc and CUDA headers, not just the runtime libraries.
FROM docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    clang \
    cmake \
    curl \
    git \
    libomp-dev \
    ninja-build \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    util-linux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY ml/requirements.txt ml/requirements.txt
RUN python3 -m venv "$VIRTUAL_ENV" \
    && pip install --no-cache-dir -r ml/requirements.txt

COPY ml/ ml/
COPY llama/flags.json llama/flags.json
COPY --chmod=755 runner.sh /usr/local/bin/runner

RUN mkdir -p /data

ENTRYPOINT ["/usr/local/bin/runner"]
CMD ["setup"]
