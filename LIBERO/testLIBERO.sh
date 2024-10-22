#!/bin/bash
#SBATCH -J LIBERO_eval
#SBATCH -t 10:00:00
#SBATCH --output=/home/tw9146/tw/LIBERO/openvla/slurm_log/slurm_out/%x_%j.out
#SBATCH --error=/home/tw9146/tw/LIBERO/openvla/slurm_log/slurm_err/%x_%j.err
#SBATCH -p tier3
#SBATCH -A domainacs
#SBATCH --mail-user=tw9146@rit.edu
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openvla_LIBERO
#sh scripts/ds_tune_vicuna.sh
cd /home/tw9146/tw/LIBERO/openvla
python experiments/robot/libero/run_libero_eval.py   --model_family openvla   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial   --task_suite_name libero_spatial   --center_crop True
