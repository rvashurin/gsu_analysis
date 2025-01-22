#!/bin/bash

#SBATCH --job-name=llama_gsu_ablation          # job name

#SBATCH --qos=cscc-gpu-qos

#SBATCH --partition=long                      # queue name

#SBATCH --mail-type=all                      # mail events (none, begin, end, fail, all)

#SBATCH --mail-user=Roman.Vashurin@mbzuai.ac.ae   # where to send mail

#SBATCH --nodes=1

#SBATCH --mem-per-cpu=100000                         # job memory request in megabytes

#SBATCH --gres=gpu:4                             # number of gpus

#SBATCH --time=03-00:00:00                   # time limit hrs:min:sec or dd-hrs:min:sec

#SBATCH --output=/l/users/maxim.panov/storage_vashurin/log/llama_gsu_ablation.out

module load anaconda3

#command part

source activate polygraph_maiya

cd /home/maxim.panov/workspace_vashurin/gsu_analysis

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HF_HOME=/l/users/maxim.panov/cache python recalc_greedy_sem_mat.py --model llama8b --datasets trivia mmlu coqa --in_dir /l/users/maxim.panov/storage_vashurin/mans --out_dir /l/users/maxim.panov/storage_vashurin/processed_mans --batch_size 5 --cuda_device 0 &
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HF_HOME=/l/users/maxim.panov/cache python recalc_greedy_sem_mat.py --model llama8b --datasets gsm8k_cot xsum --in_dir /l/users/maxim.panov/storage_vashurin/mans --out_dir /l/users/maxim.panov/storage_vashurin/processed_mans --batch_size 5 --cuda_device 0 &
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HF_HOME=/l/users/maxim.panov/cache python recalc_greedy_sem_mat.py --model llama8b --datasets wmt14_fren wmt19_deen --in_dir /l/users/maxim.panov/storage_vashurin/mans --out_dir /l/users/maxim.panov/storage_vashurin/processed_mans --batch_size 5 --cuda_device 0 &

wait
