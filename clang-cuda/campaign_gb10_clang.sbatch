#!/bin/bash
#SBATCH --job-name=gb10-clang
#SBATCH --account=dphpc
#SBATCH --gpus=gb10:1
#SBATCH --output=/scratch/users/tlemagnan/clang-campaign/jobs/logs/%x-%j.out
# clang-CUDA campaign on one GB10 node: GPU code compiled by clang 22, GPU flags varied too.
# Usage: sbatch -t HH:MM:SS [--nodelist=studgpu-sparkNN] campaign_gb10_clang.sbatch <llama|gromacs> <results-subdir> <sweep|random> [extra runner args]
#   e.g. ... llama gb10-clang-random300 random --count 300 --min-flags 1
# Benchmark only (--no-profile). Resumable: resubmit the same command to continue.
set -o pipefail
C=/scratch/users/tlemagnan/clang-campaign
APP=$1; NAME=$2; MODE=$3; shift 3
# GROMACS results get a "gromacs-" prefix so they never share a directory with llama runs
[ "$APP" = gromacs ] && NAME=gromacs-$NAME
source $C/env-gb10-clang.sh
echo "node $(hostname), $(nproc) CPUs, $APP $MODE $*, results $C/results/$NAME, start $(date)"
clang++ --version | head -1; cmake --version | head -1; nvcc --version | tail -1
/usr/bin/drop-caches 2>/dev/null || true      # GB10: free buffer cache so it can serve as VRAM (ISG docs)
if [ "$APP" = llama ]; then
    cd $C/llama
    python3 llama_gpu_bench-random.py $MODE --device cuda --cuda-arch 121 --cuda-compiler clang \
        --flags-file flags-clang.json --source $LLAMA_CLANG_SRC \
        --model /scratch/users/tlemagnan/llama-campaign/llama/model.gguf --no-profile \
        --workers $(nproc) --build-dir $TMPDIR/build --results-dir $C/results/$NAME "$@"
else
    G=/scratch/users/tlemagnan/gromacs-campaign/gromacs
    cd $C/gromacs
    # same input/steps as the old GB10 GROMACS campaign (water_0096, 50000 steps, 1 warmup + 3)
    python3 gromacs_gpu_bench-random.py $MODE --device cuda --cuda-arch 121 --cuda-compiler clang \
        --cccl $CCCL_DIR --flags-file flags-clang.json --source $G/gromacs-src \
        --tpr $G/inputs/water_0096.tpr --fftw $G/fftw-3.3.10.tar.gz --nsteps 50000 --no-profile \
        --workers $(nproc) --threads $(nproc) --build-dir $TMPDIR/build --results-dir $C/results/$NAME "$@"
fi
rc=$?
echo "done $(date), exit $rc"
exit $rc
