#!/usr/bin/env bash
set -euo pipefail

device=cpu
work=
recommended_cmake=
recommended_compiler=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) [[ $# -ge 2 && ( $2 == cpu || $2 == gpu ) ]] || { echo "Expected --device cpu|gpu" >&2; exit 1; }; device=$2; shift 2 ;;
        --workdir) [[ $# -ge 2 && -n $2 ]] || { echo "Expected --workdir PATH" >&2; exit 1; }; work=$2; shift 2 ;;
        --cmake-flags) [[ $# -ge 2 ]] || { echo "Expected --cmake-flags STRING" >&2; exit 1; }; recommended_cmake=$2; shift 2 ;;
        --compiler-flags) [[ $# -ge 2 ]] || { echo "Expected --compiler-flags STRING" >&2; exit 1; }; recommended_compiler=$2; shift 2 ;;
        *) echo "Usage: $0 [--device cpu|gpu] [--workdir PATH] [--cmake-flags STRING] [--compiler-flags STRING]" >&2; exit 1 ;;
    esac
done

commit=922f53ed33365630999867c218fefef9a3d9fd22
model=qwen2.5-3b-instruct-q4_k_m.gguf
if [[ -z $work ]]; then
    work=$(mktemp -d)
    trap 'rm -rf -- "$work"' EXIT
else
    mkdir -p -- "$work"
    work=$(cd -- "$work" && pwd -P)
fi

if [[ -e $work/llama.cpp ]]; then
    [[ -d $work/llama.cpp/.git && $(git -C "$work/llama.cpp" rev-parse HEAD) == "$commit" ]] || {
        echo "Existing llama.cpp checkout is not at $commit" >&2; exit 1;
    }
    [[ -z $(git -C "$work/llama.cpp" status --porcelain) ]] || {
        echo "Existing llama.cpp checkout has local changes" >&2; exit 1;
    }
else
    git init -q "$work/llama.cpp"
    git -C "$work/llama.cpp" remote add origin https://github.com/ggml-org/llama.cpp.git
    git -C "$work/llama.cpp" fetch --depth 1 origin "$commit"
    git -C "$work/llama.cpp" checkout --detach "$commit"
fi

cmake_args=()
read -r -a cmake_args <<< "$recommended_cmake"
if [[ -n $recommended_compiler ]]; then
    cmake_args+=("-DCMAKE_C_FLAGS=$recommended_compiler" "-DCMAKE_CXX_FLAGS=$recommended_compiler")
fi
gpu_layers=0
if [[ $device == gpu ]]; then
    cmake_args+=(-DGGML_CUDA=ON)
    gpu_layers=99
fi
rm -rf -- "$work/llama.cpp/build/CMakeCache.txt" "$work/llama.cpp/build/CMakeFiles"
cmake -S "$work/llama.cpp" -B "$work/llama.cpp/build" "${cmake_args[@]}"
cmake --build "$work/llama.cpp/build" --config Release --target llama-bench --parallel

if [[ ! -s $work/$model ]]; then
    [[ ! -e $work/$model ]] || { echo "Existing model is empty: $work/$model" >&2; exit 1; }
    download=$(mktemp "$work/.model.XXXXXX")
    if ! curl --fail --location --retry 3 -o "$download" \
        "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/$model"; then
        rm -f -- "$download"
        exit 1
    fi
    mv -- "$download" "$work/$model"
fi
"$work/llama.cpp/build/bin/llama-bench" -m "$work/$model" -ngl "$gpu_layers" -p 512 -n 128 -r 3 -o json
