# llama.cpp Vulkan build and ML image.
FROM docker.io/library/ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    clang \
    cmake \
    curl \
    git \
    glslc \
    libomp-dev \
    libvulkan-dev \
    llvm \
    mesa-vulkan-drivers \
    ninja-build \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    util-linux \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY ml/requirements.txt ml/requirements.txt
RUN python3 -m venv "$VIRTUAL_ENV" \
    && pip install --no-cache-dir -r ml/requirements.txt \
    && pip check \
    && python -c "import catboost, joblib, numpy, pandas, sklearn"

COPY ml/ ml/
COPY llama/flags.json llama/flags.json
COPY llama/llama_gpu_bench-random.py llama/llama_gpu_bench-profile.py llama/filter_llama_performance_flags.py llama/
COPY --chmod=755 runner.sh /usr/local/bin/runner

RUN mkdir -p /data

ENTRYPOINT ["/usr/local/bin/runner"]
CMD ["setup"]
