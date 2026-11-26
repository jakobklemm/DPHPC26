# llama.cpp Vulkan runs on RX 6600 XT with the GPU switches sampled

Davide, 2026-11-25/26. 500 runs (`random --count 500`, harmonic sampler, seed 20261021),
0 failed, 321 distinct configurations, 251 with at least one GPU switch on.
Folder: `ryzen_5800X-vulkan-random-gpuflags-500`.

## Why a new Vulkan run

The older Vulkan runs (`ryzen_5800X-vulkan-random-10k`, `-harmonic-500`) varied only CPU
settings: CMake switches and clang flags. With `-ngl 99` the whole model runs on the GPU, so
they are nearly flat (middle 90% spread: pp512 3%, tg128 1%). The `GGML_VK_*` switches were
never sampled, and the runner had two bugs that would have made them useless:

- llama.cpp only checks whether a `GGML_VK_*` variable *exists*, so the runner's "off" value
  `NAME=0` actually turned the switch on. Now "0" leaves the variable unset, inherited
  `GGML_VK_*` variables are removed, and `runtime_environment` in build.json records what
  was really set.
- `GGML_VK_DISABLE_COOPMAP` was a typo for `COOPMAT`. The runner now refuses any variable that
  `ggml-vulkan.cpp` does not read.

`ml/load_llama.py` now reads `runtime_environment`: a set switch is `ON`, an unset one keeps
the default (older runs stay default).

## Hardware and software

- AMD Ryzen 7 5800X + AMD Radeon RX 6600 XT (RADV NAVI23, Mesa 25.2.8), no coopmat or
  integer dot product (the Ubuntu 24.04 glslc is too old for integer dot shaders).
- Docker image `dphpc26:vulkan-fixed` toolchain (Ubuntu 24.04, clang 18.1.3) plus the fixed
  scripts. llama.cpp 922f53ed, Qwen2.5-3B-Instruct Q4_K_M, pp512 + tg128, 3 repetitions.

## Flag catalog

`flags.json` has 12 Vulkan switches. This run used `flags-vulkan-rx6600xt.json` (in the
folder): the 7 that changed speed on this card when each was tested alone. Left out:
`COOPMAT`, `DISABLE_MMVQ`, `FORCE_MMVQ`, `MULTI_ADD`, `DESCRIPTOR_REUSE` (no effect here),
and `PREFER_HOST_MEMORY` (puts the model in system RAM, about 20x slower).

Shader compiler flags were also tried (glslc `-O`, `-Os`, `-O0` through a wrapper): the
SPIR-V and the GPU machine code change, but speed stays within 0.3%, so they are not sampled.

## Results

Spread (middle 90%): pp512 93%, tg128 144%. Effect of each switch alone, relative to the
249 runs with all GPU switches off (CPU flags alone move speed by 1.2% / 0.6%):

| Only this switch on | Runs | pp512 | tg128 |
|---|---|---|---|
| `DISABLE_F16` | 19 | -47.4% | -0.1% |
| `SERIALIZE_SUBMISSIONS` | 12 | -2.3% | -51.5% |
| `DISABLE_FUSION` | 16 | -1.0% | -6.7% |
| `DISABLE_HOST_VISIBLE_VIDMEM` | 14 | 0.0% | -4.8% |
| `DISABLE_GRAPH_OPTIMIZE` | 9 | 0.0% | -0.8% |
| `DISABLE_ASYNC` | 8 | -0.2% | -0.5% |
| `ALLOW_GRAPHICS_QUEUE` | 10 | 0.0% | +0.4% |

Most of the spread comes from F16 (pp512) and SERIALIZE_SUBMISSIONS (tg128). The switches
choose GPU code paths and scheduling; they do not change how the shaders are compiled.
`ml/live_llama.py` does not pass these switches yet, so live sampling cannot reproduce them.
