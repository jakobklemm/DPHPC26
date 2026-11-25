# Toolchain for clang-CUDA runs on the RTX 5060 Ti nodes (x86_64): clang 22 builds CPU *and* GPU code.
# Usage (inside a 5060 Ti job): source /scratch/users/tlemagnan/clang-campaign/env-5060ti-clang.sh
T=/scratch/users/tlemagnan/clang-cuda-test
. /etc/profile.d/modules.sh
module load cuda/12.8                                   # same CUDA as the old nvcc 5060 Ti runs
CUDA=/cluster/data/cuda/x86_64/12.8
# cuda/12.8 module prepends a non-existent $CUDA/lib; libcudart.so.12 lives in lib64
export LD_LIBRARY_PATH=$CUDA/lib64:$T/LLVM-22.1.8-Linux-X64/lib/x86_64-unknown-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}  # + libomp
export PATH=$T/LLVM-22.1.8-Linux-X64/bin:/scratch/users/tlemagnan/clang-campaign/bin-x86:$PATH   # clang 22.1.8 (+lld cfg), CMake 3.31.6
export CUDACXX=$(command -v nvcc)                       # scripts use it to find the CUDA root
export LLAMA_CLANG_SRC=$T/llama.cpp-clang               # llama.cpp 62df1815b + llama/clang-cuda.patch
