#!/bin/bash
#SBATCH --job-name=5060ti-gmx-clang-test
#SBATCH --account=dphpc
#SBATCH --gpus=5060ti:1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/tlemagnan/clang-campaign/jobs/logs/%x-%j.out
# Test before the GROMACS clang-CUDA campaign on the RTX 5060 Ti (sm_120, Xeon E5-2680 v4, 3 CPUs per job).
# 3 builds, same benchmark as the old 5060 Ti GPU runs (water_0096, 50000 steps, 1 warmup + 3):
#   1. clang baseline   (clang 22 compiles CPU and GPU code, flags-5060ti-clang.json)
#   2. nvcc baseline    (reference: same clang 22 for CPU code, nvcc 12.8 for GPU code, flags-5060ti.json)
#   3. clang + cuda:-fno-inline-functions (GPU flags really reach the GPU code?)
# Checks: CCCL 3.4.3 + CUDA 12.8 + clang 22 compile, libomp found, build time with 3 cores, ns/day,
# energies clang vs nvcc (correctness). Results: clang-campaign/results/gromacs-5060ti-clang-test
set -o pipefail
C=/scratch/users/tlemagnan/clang-campaign
G=/scratch/users/tlemagnan/gromacs-campaign/gromacs
source $C/env-5060ti-clang.sh
export CCCL_DIR=/scratch/users/tlemagnan/clang-cuda-test/cccl-3.4.3   # CUDA's own cub does not compile with clang 22 (GB10)
OUT=$C/results/gromacs-5060ti-clang-test
BUILD=/tmp/tlemagnan-gmxclang-$SLURM_JOB_ID/build                    # node-local disk
mkdir -p $BUILD $OUT
trap "rm -rf $(dirname $BUILD)" EXIT
N=$(nproc)
echo "node $(hostname), $N CPUs, start $(date)"
lscpu | grep -E "Model name"; nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
echo "other jobs on this node: $(squeue -h -w $(hostname) | wc -l)"
clang++ --version | head -1; cmake --version | head -1; nvcc --version | tail -1
echo "libomp: $(clang++ -print-file-name=libomp.so)"
cd $C/gromacs
COMMON=(--device cuda --no-profile --source $G/gromacs-src --fftw $G/fftw-3.3.10.tar.gz
        --tpr $G/inputs/water_0096.tpr --nsteps 50000 --warmup 1 --repetitions 3
        --workers $N --threads $N --build-dir $BUILD --results-dir $OUT)
build() {   # build <label> <flags-file> <extra profile args...> -- <baseline overrides...>
    local LABEL=$1 FLAGS=$2; shift 2
    local EXTRA=() OVR=()
    while [ $# -gt 0 ] && [ "$1" != -- ]; do EXTRA+=("$1"); shift; done
    [ "$1" = -- ] && shift; OVR=("$@")
    mapfile -t ARGS < <(python3 $C/jobs/baseline_args.py $FLAGS 120 "${OVR[@]}") || { echo "args failed"; return 1; }
    echo "=== $LABEL $(date)"; printf '  %s\n' "${ARGS[@]}"
    python3 gromacs_gpu_bench-profile.py "${COMMON[@]}" --flags-file $FLAGS "${EXTRA[@]}" "${ARGS[@]}" \
        > $OUT/$LABEL.out 2>&1
    local rc=$?; echo "=== $LABEL exit $rc $(date)"; tail -3 $OUT/$LABEL.out
    grep -m1 "^Run directory:" $OUT/$LABEL.out | sed 's/^Run directory: //' > $OUT/$LABEL.dir
    return $rc
}
CLANG=(--cuda-compiler clang --cccl $CCCL_DIR)
build clang-baseline $C/gromacs/flags-5060ti-clang.json "${CLANG[@]}" --
# nvcc 12.8 does not know clang 22 as host compiler: allow it without changing the recorded options
export NVCC_APPEND_FLAGS=-allow-unsupported-compiler
build nvcc-baseline $C/gromacs/flags-5060ti.json --
unset NVCC_APPEND_FLAGS
build clang-cuda-noinline $C/gromacs/flags-5060ti-clang.json "${CLANG[@]}" -- cuda:-finline-functions -fno-inline-functions

echo "=== summary $(date)"
for L in clang-baseline nvcc-baseline clang-cuda-noinline; do
    D=$(cat $OUT/$L.dir 2>/dev/null); [ -n "$D" ] || { echo "$L: no run directory"; continue; }
    python3 - $L $D <<'EOF'
import json, sys
label, d = sys.argv[1], sys.argv[2]
b = json.load(open(d + "/build.json"))
try:
    m = json.load(open(d + "/benchmark.json"))
    perf = f"mean {m['mean']:.1f} ns/day, runs {[round(r['value'], 1) for r in m['runs']]}"
except OSError:
    perf = "no benchmark"
print(f"{label}: status {b.get('status')}, build {b.get('build_time_seconds')} s, {perf}, "
      f"cuda {b.get('build_facts', {}).get('cuda_compiler', '?')}")
if b.get("error"): print("  error:", b["error"][:300])
EOF
    # energies of the first measured run (clang vs nvcc must agree)
    L0=$D/benchmark/measured0/md.log
    [ -f $L0 ] && awk '/A V E R A G E S/{f=1} f&&/Energies/{g=1} g{print "   " $0; if(++n>8) exit}' $L0
done
echo "done $(date)"
