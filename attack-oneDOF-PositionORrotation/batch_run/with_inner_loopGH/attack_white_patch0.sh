#!/bin/bash
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
maskidx='0'
tags="LRDecay WarmUp single DoF1 InnerLoop100"
filterGripTrainTo1=false
# Load parameters from config.json
config_file="openvla-main/attack-oneDOF-PositionORrotation/batch_run/with_inner_loop/config.json"
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
dataset=$(jq -r '.dataset' $config_file)
tags="${tag_prefix} ${tags}"
# Run the Python script with parsed parameters
python openvla-main/attack-oneDOF-PositionORrotation/attack_white_patch_single_withInnerLoop.py \
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
    --innerLoop 50 \
    --dataset $dataset
