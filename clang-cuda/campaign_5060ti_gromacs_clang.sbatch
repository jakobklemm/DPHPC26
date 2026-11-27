#!/bin/bash
#SBATCH --job-name=5060ti-gmx-clang
#SBATCH --account=dphpc
#SBATCH --gpus=5060ti:1
#SBATCH --output=/scratch/users/tlemagnan/clang-campaign/jobs/logs/%x-%j.out
# GROMACS clang-CUDA campaign on one RTX 5060 Ti node (sm_120, Xeon E5-2680 v4, 3 CPUs per job):
# clang 22 compiles CPU and GPU code, GPU flags sampled too (flags-5060ti-clang.json).
# Same benchmark as the old 5060 Ti GPU runs: water_0096, 50000 steps, 1 warmup + 3 measured, GPU-resident.
# Usage: sbatch -t HH:MM:SS campaign_5060ti_gromacs_clang.sbatch <results-subdir> <sweep|random> [extra runner args]
#   e.g. ... gromacs-5060ti-clang-sweep sweep
#        ... gromacs-5060ti-clang-random random --count 100 --min-flags 1
# After the runner, one more baseline build into <results-subdir>-baseline (noise / drift check;
# the sweep has its baseline at the start). Resumable: resubmit the same command (BUILD path is fixed,
# because the runner stores build_dir in <mode>-settings.json).
set -o pipefail
C=/scratch/users/tlemagnan/clang-campaign
G=/scratch/users/tlemagnan/gromacs-campaign/gromacs
NAME=$1; MODE=$2; shift 2
source $C/env-5060ti-clang.sh
export CCCL_DIR=/scratch/users/tlemagnan/clang-cuda-test/cccl-3.4.3   # CUDA's own cub does not compile with clang 22
BUILD=/tmp/tlemagnan-gmxclang-$NAME/build                            # node-local disk, same path on every resume
mkdir -p $BUILD
trap "rm -rf $(dirname $BUILD)" EXIT
N=$(nproc)
echo "node $(hostname), $N CPUs, $MODE $*, results $C/results/$NAME, start $(date)"
echo "other jobs on this node: $(squeue -h -w $(hostname) | wc -l)"
clang++ --version | head -1; cmake --version | head -1; nvcc --version | tail -1
cd $C/gromacs
FLAGS=$C/gromacs/flags-5060ti-clang.json
COMMON=(--device cuda --cuda-compiler clang --cccl $CCCL_DIR --flags-file $FLAGS --no-profile
        --source $G/gromacs-src --fftw $G/fftw-3.3.10.tar.gz --tpr $G/inputs/water_0096.tpr --nsteps 50000
        --warmup 1 --repetitions 3 --workers $N --threads $N --build-dir $BUILD)
python3 gromacs_gpu_bench-random.py $MODE --cuda-arch 120 "${COMMON[@]}" --results-dir $C/results/$NAME "$@"
rc=$?
echo "=== $MODE exit $rc $(date); end baseline -> $C/results/$NAME-baseline"
mapfile -t ARGS < <(python3 $C/jobs/baseline_args.py $FLAGS 120)
mkdir -p $C/results/$NAME-baseline
# same host snapshot as the runner's runs (platform name for ml/load_gromacs.py)
python3 gromacs_gpu_bench-profile.py "${COMMON[@]}" --results-dir $C/results/$NAME-baseline \
    --system-info $C/results/$NAME/host-system-cuda-120.json "${ARGS[@]}"
echo "done $(date), exit $rc"
exit $rc
