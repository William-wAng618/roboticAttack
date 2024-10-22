#!/bin/bash
#SBATCH -J openvla_attack_test
#SBATCH -t 40:00:00
#SBATCH --output=/home/tw9146/tw/openvla-main/run/white_patch_attack/slurm_out/%x_%j.out
#SBATCH --error=/home/tw9146/tw/openvla-main/run/white_patch_attack/slurm_err/%x_%j.err
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
export WANDB_API_KEY='2afe2846a028d2bc651361a4fbdf85cfbd9d9eab'
wandb login
python /home/tw9146/tw/openvla-main/attack-oneDOF-rc/attack_white_patch.py --lr 1e-3 --maskidx 1