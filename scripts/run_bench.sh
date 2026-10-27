#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=jevbench
#SBATCH --exclude=gpuk001,gpuk002,gpuk003,gpuk004,gpuk005,gpuk006,gpuk007,gpuk008,gpuk009,gpuk010,gpuk011,gpuk012,gpuk013,gpuk014,gpuk015,gpuk016,gpuk017,gpuk018
#SBATCH --output=/scratch/ad2416/jevbench.out
# Score the three 1B adapters on R-Judge and Agent-SafetyBench. This is the
# benchmark criterion of the goal, and it needs no retraining: the adapters
# already exist. Runs on its own GPU while the Glimmer QLoRA job holds another.
export JEV_DATA=/scratch/ad2416/jevdata
export HF_HOME=/scratch/ad2416/hf
cd /scratch/ad2416
for s in 0 1 2; do
  echo "########## seed $s ##########"
  /scratch/ad2416/venv-torch/bin/python -u /scratch/ad2416/jevsrc/eval_bench.py \
    --adapter /scratch/ad2416/jevout-fix/adapter_seed$s \
    --bench both \
    --out /scratch/ad2416/jevout-fix/bench_seed$s.json
done
echo BENCH_DONE
