# Build-parameter ML container

The CUDA and Vulkan images combine the `ml` package (including all models in
`ml/requirements.txt`), the llama.cpp flag catalog, and clang/CMake/Ninja.
The CUDA image includes `nvcc` and CUDA development libraries; the Vulkan image
includes Vulkan headers, `glslc`, and Mesa Vulkan drivers without a CUDA toolkit.
Neither contains **profiling tools**, GGUF weights, trained `.joblib` models, a
llama.cpp checkout, or benchmark history. The existing
`llama/llama_gpu_bench-*.py` scripts require profiling and are not included as
container runners. The future predict–build–evaluate loop is not implemented.

Build locally with `docker build -t dphpc26:cuda .` or
`docker build -f Dockerfile.vulkan -t dphpc26:vulkan .`. Run the **Publish build
and ML image** workflow manually from GitHub Actions to publish
`ghcr.io/<owner>/<repo>:cuda` and `:vulkan`, plus commit-specific
`:cuda-sha-<commit>` and `:vulkan-sha-<commit>` tags (amd64 only). The existing
`:latest` and `:sha-<commit>` tags continue to point to the CUDA image. Replace
the image name below with the appropriate local or published tag.

```sh
docker run --rm -it --gpus all \
  --mount type=bind,src=/absolute/path/to/llama.cpp,dst=/workspace/llama.cpp,readonly \
  --mount type=bind,src=/absolute/path/to/gguf,dst=/workspace/gguf,readonly \
  --mount type=bind,src=/absolute/path/to/trained-models,dst=/workspace/models \
  --mount type=bind,src=/absolute/path/to/bench-results,dst=/workspace/llama/results \
  --mount type=bind,src=/absolute/path/to/build,dst=/workspace/build \
  dphpc26:cuda
```

Create the host directories before starting the container. Use the same mounts
for subsequent cycles so builds, trained models, and results survive container
restarts. The GGUF mount is for inference weights; `models/` is for ML `.joblib`
files. For CUDA runs install the NVIDIA Container Toolkit on the host and use a
compatible NVIDIA driver. For Vulkan, use `dphpc26:vulkan`, omit `--gpus all`,
and pass through the host GPU device (for example, `--device /dev/dri`) with a
compatible host driver. For CPU-only builds omit GPU device options. Native CPU
tuning sees the host CPU through the container.

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

Choose a CUDA architecture supported by your GPU/toolkit. In the Vulkan image,
replace the CUDA-specific CMake arguments with `-DGGML_CUDA=OFF -DGGML_VULKAN=ON`;
for CPU builds use `-DGGML_CUDA=OFF` and omit the CUDA-specific arguments.
Reconfigure the same build directory when flags change; use a fresh build
directory if you change compilers or generators.
`llama/flags.json` lists tuning options, and
`python llama/filter_llama_performance_flags.py --help` lists the host-filtering
options. `llama-bench` alone does not produce the full run metadata expected by
`ml.load`; producing those records belongs to the future runner.
