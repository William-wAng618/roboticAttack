#!/bin/bash
export WANDB_API_KEY='2afe2846a028d2bc651361a4fbdf85cfbd9d9eab'
wandb login


# specify parameters
geometry=false
colorjitter=false
maskidx='0,1,2,3,4,5'
tags="LRDecay WarmUp DoFALL Untargeted-012345"
filterGripTrainTo1=false
lr=2e-3
device=3
iter=100000
accumulate=1
bs=8
warmup=200
patch_size="3,50,50"
wandb_project="openvla_white_patch_v2"

servername="A600"
# 初始化另一个字符串的变量
if [ "$servername" == "rc" ]; then
    server=/home/tw9146/tw
    cd /home/tw9146/tw
else
    server=/spl_data/tw9146
    cd /spl_data/tw9146
fi


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
    --wandb_project $wandb_project
