#!/bin/bash
#SBATCH --job-name=5060ti-gmx-setup
#SBATCH --account=dphpc
#SBATCH --gpus=5060ti:1
#SBATCH --nodelist=studgpu-node01
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/users/tlemagnan/gromacs-campaign/jobs/logs/%x-%j.out
# One-time check on the RTX 5060 Ti node (x86): hardware, perf, toolchain, one baseline
# CUDA build (sm_120), then timing of GPU-resident and CPU-only mdrun to choose nsteps.
set -o pipefail
G=/scratch/users/tlemagnan/gromacs-campaign/gromacs
source /scratch/users/tlemagnan/llama-campaign/env.sh
W=/scratch/users/tlemagnan/gromacs-campaign   # env.sh sets W to llama-campaign
N=$(nproc)
echo "node $(hostname), $N CPUs, start $(date)"
echo "== hardware"; lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|L3|Flags"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
free -g | head -2
echo "perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)"
command -v perf && perf stat -e cycles,instructions true 2>&1 | tail -6
echo "== tools"; clang --version | head -1; nvcc --version | tail -2; cmake --version | head -1
echo 'int main(){return 0;}' > $TMPDIR/x.c
clang -march=native -### -c $TMPDIR/x.c -o /dev/null 2>&1 | grep -o '"-target-cpu" "[^"]*"'
for f in -march=x86-64-v3 -march=x86-64-v4 -march=native -flto=thin; do
  clang $f $TMPDIR/x.c -o $TMPDIR/x && $TMPDIR/x && echo "ok $f" || echo "FAIL $f"; done
cd $G
python3 filter_gromacs_performance_flags.py --device cuda --input flags-5060ti.json --output $TMPDIR/filtered.json; cat $TMPDIR/filtered.json 2>&1 | head -80

echo "== baseline build + GPU and CPU-only benchmark (short) $(date)"
python3 gromacs_gpu_bench-profile.py --device cuda --no-profile --workers $N --threads $N \
    --flags-file flags-5060ti.json --build-dir $TMPDIR/build --results-dir $W/results/setup-5060ti \
    --cmake-arg GMX_GPU=CUDA --cmake-arg CMAKE_CUDA_ARCHITECTURES=120 --cmake-arg GMX_SIMD=AVX2_256 \
    --tpr $G/inputs/water_0096.tpr --nsteps 10000 --warmup 0 --repetitions 1 \
    --cpu-tpr $G/inputs/water_0048.tpr --cpu-nsteps 1000
GMX=$TMPDIR/build/bin/gmx
$GMX --version | grep -E "version|Precision|SIMD|FFT|GPU support|CUDA|C\+\+ compiler flags"

echo "== timing per input $(date)"
for SIZE in 0048 0096; do for MODE in gpu cpu; do
  if [ $MODE = gpu ]; then A="-nb gpu -pme gpu -bonded cpu -update gpu"; S=10000; E=""
  else A="-nb cpu -pme cpu -bonded cpu -update cpu"; S=2000; E="GMX_DISABLE_GPU_DETECTION=1"; fi
  mkdir -p $TMPDIR/t$SIZE$MODE && cd $TMPDIR/t$SIZE$MODE
  env $E GMX_MAXBACKUP=-1 $GMX mdrun -s $G/inputs/water_$SIZE.tpr -ntmpi 1 -ntomp $N -pin on \
      $A -nsteps $S -resethway -notunepme -noconfout -g md.log > mdrun.out 2>&1 || tail -20 mdrun.out
  echo "water $SIZE $MODE ($S steps): $(grep -E '^Performance:' md.log) / $(grep -E '^ +Time:' md.log)"
  cd $G
done; done
echo "done $(date)"
