#!/bin/bash
#SBATCH --job-name=gb10-gromacs
#SBATCH --account=dphpc
#SBATCH --gpus=gb10:1
#SBATCH --nodelist=studgpu-spark01
#SBATCH --output=/scratch/users/tlemagnan/gromacs-campaign/jobs/logs/%x-%j.out
# GROMACS campaign on one GB10 node (pinned to studgpu-spark01 like the llama runs, so results are comparable).
# Usage: sbatch -t HH:MM:SS campaign_gb10.sbatch <results-subdir> <sweep|subset> [extra runner args]
#   e.g. campaign_gb10.sbatch gb10-subset100 subset --count 100
# Benchmark only (--no-profile): no perf, no HPCToolkit.
# Input + steps from setup job 2146: water 96k atoms ~227 ns/day -> 50000 steps ~40 s per mdrun.
# Resumable: resubmitting the same command skips completed configs, retries failed ones.
set -o pipefail
W=/scratch/users/tlemagnan/gromacs-campaign
G=$W/gromacs
TPR=$G/inputs/water_0096.tpr
NSTEPS=50000
NAME=$1; MODE=$2; shift 2
source /scratch/users/tlemagnan/llama-campaign/env-gb10.sh
W=/scratch/users/tlemagnan/gromacs-campaign   # env-gb10.sh sets W to llama-campaign
echo "node $(hostname), $(nproc) CPUs, mode $MODE $*, results $W/results/$NAME, start $(date)"
/usr/bin/drop-caches 2>/dev/null || true      # GB10: free buffer cache so it can serve as VRAM (ISG docs)
cd $G
python3 gromacs_gpu_bench-random.py $MODE --device cuda --cuda-arch 121 --no-profile \
    --tpr $TPR --nsteps $NSTEPS --workers $(nproc) --threads $(nproc) \
    --build-dir $TMPDIR/build --results-dir $W/results/$NAME "$@"
rc=$?
echo "done $(date), exit $rc"
exit $rc
