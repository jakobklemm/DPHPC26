# Build-parameter ML container

The image combines the `ml` package (including all models in `ml/requirements.txt`),
the llama.cpp flag catalog, and clang/CMake/Ninja, CUDA (`nvcc`), and Vulkan build
dependencies. It contains **no profiling tools**, GGUF weights, trained `.joblib`
models, llama.cpp checkout, or benchmark history. The existing
`llama/llama_gpu_bench-*.py` scripts require profiling and are not included as
container runners. The future predict–build–evaluate loop is not implemented.

Build locally with `docker build -t dphpc26:local .`, or run the **Publish build
and ML image** workflow manually from GitHub Actions to publish
`ghcr.io/<owner>/<repo>:latest` and `:sha-<commit>` (amd64 only). Replace the
image name below with either your local tag or the published tag.

```sh
docker run --rm -it --gpus all \
  --mount type=bind,src=/absolute/path/to/llama.cpp,dst=/workspace/llama.cpp,readonly \
  --mount type=bind,src=/absolute/path/to/gguf,dst=/workspace/gguf,readonly \
  --mount type=bind,src=/absolute/path/to/trained-models,dst=/workspace/models \
  --mount type=bind,src=/absolute/path/to/bench-results,dst=/workspace/llama/results \
  --mount type=bind,src=/absolute/path/to/build,dst=/workspace/build \
  dphpc26:local
```

Create the host directories before starting the container. Use the same mounts
for subsequent cycles so builds, trained models, and results survive container
restarts. The GGUF mount is for inference weights; `models/` is for ML `.joblib`
files. For CPU-only builds omit `--gpus all`; for CUDA runs install the NVIDIA
Container Toolkit on the host and use a compatible NVIDIA driver. Vulkan also
requires access to the host GPU device/driver (for example `/dev/dri`). Native
CPU tuning sees the host CPU through the container.

Inside the container, the ML CLI works from `/workspace`:

```sh
python -m ml.main train --app llama.cpp --model random-forest
python -m ml.main predict --app llama.cpp --model random-forest \
  --system /workspace/llama/results/<run>/system.json
python -m ml.main suggest --app llama.cpp --model random-forest \
  --system /workspace/llama/results/<run>/system.json -n 5
```

`train` writes to `/workspace/models` by default; it reads mounted history from
`/workspace/llama/results` (or pass `--data` to specify another folder/ZIP).
No history or trained model is required just to build llama.cpp. For example,
build `llama-bench` without profiling, reusing the mounted build directory:

```sh
cmake -S /workspace/llama.cpp -B /workspace/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_HOST_COMPILER=clang++
cmake --build /workspace/build --target llama-bench -j 4
/workspace/build/bin/llama-bench -m /workspace/gguf/model.gguf
```

Choose a CUDA architecture supported by your GPU/toolkit; for CPU builds use
`-DGGML_CUDA=OFF` and omit the CUDA-specific CMake arguments. For Vulkan builds,
use `-DGGML_VULKAN=ON` instead. Reconfigure the same build directory when flags
change; use a fresh build directory if you change compilers or generators.
`llama/flags.json` lists tuning options, and
`python llama/filter_llama_performance_flags.py --help` lists the host-filtering
options. `llama-bench` alone does not produce the full run metadata expected by
`ml.load`; producing those records belongs to the future runner.
