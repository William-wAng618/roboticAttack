#!/bin/bash
#SBATCH -J openvla_inference_test
#SBATCH -t 40:00:00
#SBATCH --output=./output/%x_%j.out
#SBATCH --error=./output/%x_%j.err
#SBATCH -p tier3
#SBATCH -A domainacs
#SBATCH --mail-user=tw9146@rit.edu
#SBATCH --mem=20G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#python scripts/find_no_image.py
#spack load cuda@12.3.0/goxbmvf
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openvla
#sh scripts/ds_tune_vicuna.sh
python 0testdemo/openvla_inference_test2.py