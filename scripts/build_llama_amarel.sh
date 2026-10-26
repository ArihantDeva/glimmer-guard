#!/usr/bin/env bash
# Build llama.cpp with CUDA on Amarel. Submitted as a batch job, never run on
# the login node.
#
# Three cluster facts this encodes, each of which cost an attempt:
#   1. compute nodes have NO git and NO cmake (gcc/g++/make/curl only), so the
#      source comes from a tarball and cmake is installed into a venv via uv
#   2. `module load hpc_sdk` puts nvc++ ahead of gcc and CMake then picks it as
#      the host compiler; nvc++ rejects gcc-only warning flags and the build
#      dies inside ggml-base. The host compilers are pinned explicitly.
#   3. the legacy top-level Makefile is a stub that tells you to use CMake
#
# Usage: bash scripts/build_llama_amarel.sh   (prints the job id)
set -euo pipefail
HOST="${JEV_HOST:-amarel}"
SCRATCH="${JEV_SCRATCH:-/scratch/ad2416}"
VENV="${JEV_VENV:-$SCRATCH/venv-glimmer}"

ssh "$HOST" "cat > $SCRATCH/build_llama.sh <<'SBATCH'
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpuk001,gpuk002,gpuk003,gpuk004,gpuk005,gpuk006,gpuk007,gpuk008,gpuk009,gpuk010,gpuk011,gpuk012,gpuk013,gpuk014,gpuk015,gpuk016,gpuk017,gpuk018
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=llamabuild
#SBATCH --output=$SCRATCH/llamabuild.out
source /etc/profile.d/modules.sh 2>/dev/null || true
module load hpc_sdk/26.3
export PATH=/usr/bin:/bin:/usr/local/bin:\$PATH
CMAKE=$VENV/bin/cmake
cd $SCRATCH
rm -rf llamasrc && mkdir -p llamasrc && cd llamasrc
curl -sL https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.tar.gz -o l.tgz || exit 1
tar xzf l.tgz || exit 1
cd llama.cpp-master
\$CMAKE -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_UI_BUILD_ASSETS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=\"80;86;89\" || exit 2
\$CMAKE --build build --target llama-server llama-cli llama-quantize -j 16 || exit 3
echo BUILD_OK
SBATCH
chmod +x $SCRATCH/build_llama.sh && sbatch --parsable $SCRATCH/build_llama.sh"
