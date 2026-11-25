#!/bin/bash
#SBATCH --job-name=5060ti-clang
#SBATCH --account=dphpc
#SBATCH --gpus=5060ti:1
#SBATCH --output=/scratch/users/tlemagnan/clang-campaign/jobs/logs/%x-%j.out
# llama.cpp clang-CUDA campaign on one RTX 5060 Ti node (sm_120, host Xeon E5-2680 v4, 3 CPUs per job),
# The cluster ignores --nodelist/--exclude: jobs land on the always-on node studgpu-node01 (shared with others). Same scripts/catalog as the
# GB10 clang campaign (branch clang-cuda): clang 22 compiles CPU and GPU code, GPU flags are sampled too.
# Usage: sbatch -t HH:MM:SS campaign_5060ti_clang.sbatch <results-subdir> <sweep|random> [extra runner args]
# Chain (sweep, then random) in one job: campaign_5060ti_clang.sbatch chain <random-count>
# Benchmark only (--no-profile; perf blocked). llama-bench -t = our CPUs (default 28 = whole node: with GGML_OPENMP=OFF tg128 ~70 instead of ~155 and noisy, tests 5386/5390).
# Resumable: resubmit the same command.
set -o pipefail
C=/scratch/users/tlemagnan/clang-campaign
source $C/env-5060ti-clang.sh
# BUILD can be set from outside: the runner stores build_dir in <mode>-settings.json, so a resume must reuse the
# first job's path, e.g. BUILD=/tmp/tlemagnan-clang-5393/build sbatch --export=ALL ...
BUILD=${BUILD:-/tmp/tlemagnan-clang-$SLURM_JOB_ID/build}   # node-local disk
mkdir -p $BUILD
trap "rm -rf $(dirname $BUILD)" EXIT
echo "node $(hostname), $(nproc) CPUs, args $*, start $(date)"
clang++ --version | head -1; cmake --version | head -1; nvcc --version | tail -1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
run() {   # run <results-subdir> <mode> [args]
    local NAME=$1 MODE=$2; shift 2
    echo "=== $MODE -> $C/results/$NAME ($*) $(date)"
    (cd $C/llama && python3 llama_gpu_bench-random.py $MODE --device cuda --cuda-arch 120 --cuda-compiler clang \
        --flags-file flags-clang.json --source $LLAMA_CLANG_SRC \
        --model /scratch/users/tlemagnan/llama-campaign/llama/model.gguf --no-profile \
        --workers $(nproc) --threads $(nproc) --build-dir $BUILD --results-dir $C/results/$NAME "$@")
}
if [ "$1" = chain ]; then
    run 5060ti-clang-sweep sweep; rc1=$?
    run 5060ti-clang-random random --count $2 --min-flags 1; rc2=$?
    echo "chain done $(date): sweep exit $rc1, random exit $rc2"; rc=$(( rc1 || rc2 ))
else
    run "$@"; rc=$?
fi
echo "done $(date), exit $rc"
exit $rc
