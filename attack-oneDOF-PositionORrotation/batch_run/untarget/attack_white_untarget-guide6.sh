#!/bin/bash
#SBATCH -J openvla_attack_test
#SBATCH -t 24:00:00
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

servername="rc"
# 初始化另一个字符串的变量
if [ "$servername" == "rc" ]; then
    server=/home/tw9146/tw
    cd /home/tw9146/tw
else
    server=/spl_data/tw9146
    cd /spl_data/tw9146
fi

# specify parameters
geometry=true
colorjitter=false
maskidx='6'
tags="LRDecay WarmUp single DoF7 Untargeted-guide"
filterGripTrainTo1=false
# Load parameters from config.json
config_file="openvla-main/attack-oneDOF-PositionORrotation/batch_run/untarget/config.json"
lr=$(jq -r '.lr' $config_file)
server=$(jq -r '.server' $config_file)
device=$(jq -r '.device' $config_file)
iter=$(jq -r '.iter' $config_file)
accumulate=$(jq -r '.accumulate' $config_file)
bs=$(jq -r '.bs' $config_file)
warmup=$(jq -r '.warmup' $config_file)
patch_size=$(jq -r '.patch_size' $config_file)
wandb_project=$(jq -r '.wandb_project' $config_file)
tag_prefix=$(jq -r '.tag_prefix' $config_file)
tags="${tag_prefix} ${tags}"
# Run the Python script with parsed parameters
python openvla-main/attack-oneDOF-PositionORrotation/attack_white_patch_untarget.py \
    --lr $lr \
    --maskidx $maskidx \
    --server $server \
    --device $device \
    --iter $iter \
    --accumulate $accumulate \
    --bs $bs \
    --warmup $warmup \
    --tags $tags \
    --filterGripTrainTo1 $filterGripTrainTo1 \
    --geometry $geometry \
    --colorjitter $colorjitter \
    --patch_size $patch_size \
    --wandb_project $wandb_project \
    --guide true
