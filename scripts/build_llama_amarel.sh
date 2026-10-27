#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpuk001,gpuk002,gpuk003,gpuk004,gpuk005,gpuk006,gpuk007,gpuk008,gpuk009,gpuk010,gpuk011,gpuk012,gpuk013,gpuk014,gpuk015,gpuk016,gpuk017,gpuk018
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=llamabuild
#SBATCH --output=/scratch/ad2416/llamabuild.out

# hpc_sdk/26.3 ships CUDA 13.1, but the node runtime is CUDA 12
# (libcudart.so.12), so the server link failed with:
#   undefined reference to `cudaGraphLaunch@libcudart.so.12'
# hpc_sdk/23.11 gives CUDA 12.3 and matches the installed runtime.
source /etc/profile.d/modules.sh 2>/dev/null || true
module load hpc_sdk/23.11
export PATH=/usr/bin:/bin:/usr/local/bin:$PATH
CMAKE=/scratch/ad2416/venv-glimmer/bin/cmake

cd /scratch/ad2416 || exit 1
rm -rf llamasrc
mkdir -p llamasrc && cd llamasrc || exit 1
curl -sL https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.tar.gz -o l.tgz || exit 1
tar xzf l.tgz || exit 1
cd llama.cpp-master || exit 1

$CMAKE -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_UI_BUILD_ASSETS=OFF \
  # Match the ACTUAL partition, not the newest card on it. Building only 80;86
  # and then landing on volta (sm_70) produced a runtime CUDA error inside
  # ggml_cuda_kernel_can_use_pdl -- programmatic dependent launch does not exist
  # before Hopper, and the binary carried no sm_70 code to fall back to. Amarel's
  # gpu partition has V100 (70), A100 (80) and L40S (89) nodes, so all are built.
  -DCMAKE_CUDA_ARCHITECTURES="70;80;89" || exit 2

$CMAKE --build build --target llama-server llama-cli llama-quantize -j 16 || exit 3
echo BUILD_OK
