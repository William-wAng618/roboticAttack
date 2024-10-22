#!/bin/bash
#SBATCH -J openvla_attack_test
#SBATCH -t 40:00:00
#SBATCH --output=/home/tw9146/tw/openvla-main/run/white_attack/slurm_out/%x_%j.out
#SBATCH --error=/home/tw9146/tw/openvla-main/run/white_attack/slurm_err/%x_%j.err
#SBATCH -p tier3
#SBATCH -A domainacs
#SBATCH --mail-user=tw9146@rit.edu
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openvla
#sh scripts/ds_tune_vicuna.sh
bash /home/tw9146/tw/openvla-main/attack/attack.sh