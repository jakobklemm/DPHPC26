#!/bin/bash
#SBATCH --job-name=gb10-gmx-setup
#SBATCH --account=dphpc
#SBATCH --gpus=gb10:1
#SBATCH --nodelist=studgpu-spark01
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/users/tlemagnan/gromacs-campaign/jobs/logs/%x-%j.out
# One-time GROMACS setup on the GB10 node (same node as the llama runs):
#  1. tool checks (clang CPU names, -mcpu=native, nvcc, SVE length)
#  2. one baseline CUDA build of GROMACS 2026.4 (= first config of the sweep)
#  3. benchmark inputs (water boxes 48k / 96k / 192k atoms) + timing to choose size and nsteps
set -o pipefail
W=/scratch/users/tlemagnan/gromacs-campaign
G=$W/gromacs
source /scratch/users/tlemagnan/llama-campaign/env-gb10.sh
W=/scratch/users/tlemagnan/gromacs-campaign   # env-gb10.sh sets W to llama-campaign
echo "node $(hostname), $(nproc) CPUs, start $(date)"
/usr/bin/drop-caches 2>/dev/null || true
echo "== tools"; clang --version | head -1; nvcc --version | tail -2; cmake --version | head -1
echo "SVE vector length (bytes): $(cat /proc/sys/abi/sve_default_vector_length)"
lscpu | grep -E "Model name|Flags"
clang --print-supported-cpus 2>&1 | grep -E "cortex-x925|cortex-a725|neoverse-v2|neoverse-n1" | tr -s " "
echo 'int main(){return 0;}' > $TMPDIR/x.c
clang -mcpu=native -### -c $TMPDIR/x.c -o /dev/null 2>&1 | grep -o '"-target-cpu" "[^"]*"'
for f in -mcpu=cortex-x925 -mcpu=neoverse-v2 -flto=thin -flto=full; do
  clang $f $TMPDIR/x.c -o $TMPDIR/x && echo "ok $f" || echo "FAIL $f"; done

echo "== baseline build $(date)"
cd $G
python3 gromacs_gpu_bench-profile.py --device cuda --no-profile --workers $(nproc) \
    --build-dir $TMPDIR/build --results-dir $W/results/setup-baseline \
    --cmake-arg GMX_GPU=CUDA --cmake-arg CMAKE_CUDA_ARCHITECTURES=121 --cmake-arg GMX_SIMD=ARM_SVE \
    --build-only
GMX=$TMPDIR/build/bin/gmx
$GMX --version | grep -E "version|Precision|SIMD|FFT|GPU support|CUDA|C\+\+ compiler flags"

echo "== inputs + timing $(date)"; mkdir -p $G/inputs
for SIZE in 0048 0096 0192; do
  ./prepare_input.sh $GMX $SIZE $G/inputs/water_$SIZE.tpr
  mkdir -p $TMPDIR/t$SIZE && cd $TMPDIR/t$SIZE
  GMX_MAXBACKUP=-1 $GMX mdrun -s $G/inputs/water_$SIZE.tpr -ntmpi 1 -ntomp $(nproc) -pin on \
      -nb gpu -pme gpu -bonded cpu -update gpu -nsteps 10000 -resethway -notunepme -noconfout \
      -g md.log > mdrun.out 2>&1 || tail -20 mdrun.out
  echo "water $SIZE: $(grep -E '^Performance:' md.log) / wall $(grep -E '^ +Time:' md.log)"
  cd $G
done
echo "done $(date)"
