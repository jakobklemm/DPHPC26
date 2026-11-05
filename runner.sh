#!/usr/bin/env bash
set -euo pipefail

readonly LLAMA_COMMIT=922f53ed33365630999867c218fefef9a3d9fd22
readonly LLAMA_REPO=https://github.com/ggml-org/llama.cpp.git
readonly GGUF_NAME=qwen2.5-3b-instruct-q4_k_m.gguf
readonly GGUF_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/$GGUF_NAME"
readonly DATA_DIR="${DATA_DIR:-/data}"
export DATA_DIR

usage() {
    cat <<'EOF'
Usage: runner [setup [options] | train [options] | active-sampling]

setup (default): discover the system, clone llama.cpp, download weights, then train.
  --skip-system      Do not generate system.json
  --skip-clone       Do not clone or check the llama.cpp checkout
  --skip-download    Do not download GGUF weights
  --skip-train       Do not train the ML model
  --model NAME       ML model to train (default: hardware; see ml.main train --help)
  -h, --help         Show this help

Mount one persistent folder at /data (override with DATA_DIR), containing the
llama/ and gromacs/ folders from this repository (including their flags.json):
  llama.cpp/         Pinned source checkout
  gguf/             Downloaded inference weights
  llama/results/    Training data supplied by the user (folders and/or ZIPs)
  models/           Trained ML .joblib files
  build/            Reserved for subsequent builds
  system.json       Container-visible hardware and toolchain snapshot for ML commands

train [ML options] trains llama.cpp using /data/llama/results, saving to /data/models.
active-sampling is a placeholder; it currently does nothing.
EOF
}

die() {
    echo "runner: $*" >&2
    exit 1
}

clone_llama() {
    local checkout="$DATA_DIR/llama.cpp"
    if [[ -e "$checkout" ]]; then
        [[ -d "$checkout/.git" ]] || die "$checkout already exists and is not a Git checkout"
        [[ "$(git -C "$checkout" rev-parse HEAD)" == "$LLAMA_COMMIT" ]] ||
            die "$checkout is not at $LLAMA_COMMIT; use --skip-clone to leave it unchanged"
        [[ -z "$(git -C "$checkout" status --porcelain)" ]] ||
            die "$checkout has local changes; use --skip-clone to leave it unchanged"
        echo "Reusing llama.cpp at $LLAMA_COMMIT"
        return
    fi

    # Publish only a complete checkout, so a failed fetch can be retried safely.
    clone_tmp=$(mktemp -d "$DATA_DIR/.llama.cpp.XXXXXX")
    trap '[[ -z "$clone_tmp" ]] || rm -rf -- "$clone_tmp"' EXIT
    git init -q "$clone_tmp"
    git -C "$clone_tmp" remote add origin "$LLAMA_REPO"
    git -C "$clone_tmp" fetch --depth 1 origin "$LLAMA_COMMIT"
    git -C "$clone_tmp" checkout --detach "$LLAMA_COMMIT"
    mv -- "$clone_tmp" "$checkout"
    clone_tmp=
}

download_gguf() {
    local weights="$DATA_DIR/gguf/$GGUF_NAME"
    mkdir -p "$DATA_DIR/gguf"
    if [[ -s "$weights" ]]; then
        echo "Reusing $weights"
        return
    fi
    # Interrupted downloads remain separate from the completed model and resume.
    curl --fail --location --retry 3 --continue-at - \
        --output "$weights.part" "$GGUF_URL"
    [[ -s "$weights.part" ]] || die "downloaded GGUF file is empty"
    mv -- "$weights.part" "$weights"
}

setup() {
    local system=true clone=true download=true train=true model=hardware
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --skip-system) system=false ;;
            --skip-clone) clone=false ;;
            --skip-download) download=false ;;
            --skip-train) train=false ;;
            --model)
                [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "--model requires a name"
                model=$2
                shift
                ;;
            -h|--help) usage; return ;;
            *) die "unknown setup option: $1" ;;
        esac
        shift
    done

    if "$train" && [[ ! -d "$DATA_DIR/llama/results" ]]; then
        die "training data missing: place llama/results in $DATA_DIR, or use --skip-train"
    fi
    mkdir -p "$DATA_DIR" "$DATA_DIR/build"
    if "$system"; then python -m ml.system --output "$DATA_DIR/system.json"; fi
    if "$clone"; then clone_llama; fi
    if "$download"; then download_gguf; fi
    if "$train"; then
        python -m ml.main --models "$DATA_DIR/models" train \
            --app llama.cpp --model "$model" --data "$DATA_DIR/llama/results"
    fi
}

mode=${1:-setup}
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
    setup) setup "$@" ;;
    train) exec python -m ml.main --models "$DATA_DIR/models" train --app llama.cpp --data "$DATA_DIR/llama/results" "$@" ;;
    active-sampling) : ;;
    -h|--help) usage ;;
    *) usage >&2; die "unknown mode: $mode" ;;
esac
