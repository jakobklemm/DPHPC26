# Toolchain for clang-CUDA runs on the GB10 (ARM, aarch64) nodes: clang 22 builds CPU *and* GPU code.
# Usage (inside a GB10 job): source /scratch/users/tlemagnan/clang-campaign/env-gb10-clang.sh
T=/scratch/users/tlemagnan/clang-cuda-test
. /etc/profile.d/modules.sh
module load cuda/13.1                                   # 13.1.1, same CUDA as the old nvcc GB10 runs
export PATH=$T/LLVM-22.1.8-Linux-ARM64/bin:$T/cmake-3.31.6-linux-aarch64/bin:$PATH   # clang 22.1.8, CMake 3.31.6
export LD_LIBRARY_PATH=$T/LLVM-22.1.8-Linux-ARM64/lib/aarch64-unknown-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}  # libomp
export CUDACXX=$(command -v nvcc)                       # scripts use it to find the CUDA root
export LLAMA_CLANG_SRC=$T/llama.cpp-clang               # llama.cpp 62df1815b + llama/clang-cuda.patch
export CCCL_DIR=$T/cccl-3.4.3                           # cub for GROMACS (CUDA 13.1's own does not compile with clang 22)
