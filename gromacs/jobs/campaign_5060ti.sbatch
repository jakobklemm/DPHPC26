#!/bin/bash
#SBATCH --job-name=5060ti-gromacs
#SBATCH --account=dphpc
#SBATCH --gpus=5060ti:1
#SBATCH --nodelist=studgpu-node01
#SBATCH --output=/scratch/users/tlemagnan/gromacs-campaign/jobs/logs/%x-%j.out
# GROMACS campaign on one RTX 5060 Ti node (x86, pinned to studgpu-node01).
# Usage: sbatch -t HH:MM:SS campaign_5060ti.sbatch <results-subdir> <sweep|subset> [extra runner args]
# Every build is measured twice with the same binary:
#   benchmark.json     GPU-resident mdrun (-nb gpu -pme gpu -update gpu), same as GB10
#   benchmark-cpu.json CPU-only mdrun (-nb cpu -pme cpu -update cpu), CPU build flags matter here
# Benchmark only (--no-profile). Resumable: resubmit the same command.
set -o pipefail
W=/scratch/users/tlemagnan/gromacs-campaign
G=$W/gromacs
TPR=${TPR:-$G/inputs/water_0096.tpr}; NSTEPS=${NSTEPS:-50000}
CPU_TPR=${CPU_TPR:-$G/inputs/water_0048.tpr}; CPU_NSTEPS=${CPU_NSTEPS:-2000}
NAME=$1; MODE=$2; shift 2
source /scratch/users/tlemagnan/llama-campaign/env.sh
W=/scratch/users/tlemagnan/gromacs-campaign   # env.sh sets W to llama-campaign
echo "node $(hostname), $(nproc) CPUs, mode $MODE $*, results $W/results/$NAME, start $(date)"
echo "GPU: $TPR $NSTEPS steps; CPU-only: $CPU_TPR $CPU_NSTEPS steps"
cd $G
python3 gromacs_gpu_bench-random.py $MODE --device cuda --cuda-arch 120 --no-profile \
    --flags-file $G/flags-5060ti.json \
    --tpr $TPR --nsteps $NSTEPS --cpu-tpr $CPU_TPR --cpu-nsteps $CPU_NSTEPS \
    --workers $(nproc) --threads $(nproc) \
    --build-dir $TMPDIR/build --results-dir $W/results/$NAME "$@"
rc=$?
echo "done $(date), exit $rc"
exit $rc
