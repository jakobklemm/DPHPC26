FROM docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04 AS hpctoolkit-layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential bzip2 ca-certificates curl file git gzip patch python3 tar unzip xz-utils \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch v1.0.0 https://github.com/spack/spack.git /opt/spack
RUN git clone --depth 1 --branch v2026.03.0 https://github.com/spack/spack-packages.git /opt/spack-packages \
    && /opt/spack/bin/spack repo set --destination /opt/spack-packages builtin
RUN /opt/spack/bin/spack compiler find \
    && /opt/spack/bin/spack external find cuda \
    && /opt/spack/bin/spack config add 'packages:cuda:buildable:false' \
    && /opt/spack/bin/spack spec 'hpctoolkit@2025.1.2+cuda~viewer ^cuda@12.8' \
    && /opt/spack/bin/spack install 'hpctoolkit@2025.1.2+cuda~viewer ^cuda@12.8' \
    && ln -s "$(/opt/spack/bin/spack location -i hpctoolkit@2025.1.2)" /opt/spack/hpctoolkit

FROM docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04 as cuda-12.8-layer
FROM docker.io/local/xaas:llama.cpp-source-x86_64 AS SOURCE
SHELL ["/bin/bash", "-c"]

ARG nproc

COPY --from=cuda-12.8-layer /usr/local/cuda-12.8 /usr/local/cuda-12.8

ENV PATH="/usr/local/cuda-12.8/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH"
RUN cmake -B build -DGGML_CUDA=ON -DGGML_OPENMP=ON -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CUDA_ARCHITECTURES="75" -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS=-g -DCMAKE_CXX_FLAGS=-g -DCMAKE_CUDA_FLAGS=-lineinfo -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_SSE42=ON \
                && cmake --build build --config Release -j ${nproc}

CMD ["/bin/bash"]

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
COPY --from=hpctoolkit-layer /opt/spack /opt/spack
ENV HPCTOOLKIT=/opt/spack/hpctoolkit
ENV PATH="/opt/spack/hpctoolkit/bin:${PATH}"
RUN apt-get update && apt-get install -y --no-install-recommends linux-tools-generic python3 && rm -rf /var/lib/apt/lists/* && ln -s "$(ls /usr/lib/linux-tools/*/perf | sort -V | tail -1)" /usr/local/bin/perf
